"""
Robust Keyframe Extraction (single-video)
- Loads preprocessed artifacts (frames, keypoints, flow_magnitudes, masks, attention_mask)
- Computes per-frame features:
    - flow median inside mask
    - pose velocity (mean joint displacement)
    - mean hand motion angle (stable via avg unit vector)
    - frame absolute diff mean
- Performs PCA (optional), KMeans medoid overcluster selection, peak detection on fused score
- Temporal greedy selection for diversity + pose confidence filtering
- Saves keyframes, key indices, QC JSON, and a preview image grid
"""
import os
import argparse
import json
import math
import numpy as np
import imageio
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ---------------------------
# Utilities
# ---------------------------

def load_frames(video_dir):
    frames_npy = os.path.join(video_dir, "frames.npy")
    frames_dir = os.path.join(video_dir, "frames_resized")
    if os.path.isfile(frames_npy):
        frames = np.load(frames_npy)
        if frames.dtype != np.uint8:
            # convert to uint8 range if floats 0..1
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)
        return frames
    if os.path.isdir(frames_dir):
        files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        imgs = []
        for f in files:
            im = imageio.imread(os.path.join(frames_dir, f))
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            if im.shape[2] == 4:
                im = im[...,:3]
            imgs.append(im)
        frames = np.stack(imgs, axis=0).astype(np.uint8)
        return frames
    raise FileNotFoundError("No frames found. Provide frames.npy or frames_resized/ in video_dir.")

def load_optional_npy(path):
    return np.load(path) if os.path.isfile(path) else None

def safe_median_masked(arr, mask):
    """Return median of arr where mask True; fallback to global median"""
    if mask is None or mask.sum() == 0:
        return float(np.median(arr))
    vals = arr[mask.astype(bool)]
    if vals.size == 0:
        return float(np.median(arr))
    return float(np.median(vals))

def compute_flow_feature_per_frame(flow_mags, masks):
    """Compute a robust per-frame flow scalar: median flow inside mask.
       flow_mags: (T-1, H, W)
       masks: (T, H, W) or None
       returns length T (pad last by repeating)"""
    if flow_mags is None:
        return None
    Tm = flow_mags.shape[0]
    T = Tm + 1
    flow_feats = np.zeros(Tm, dtype=np.float32)
    for t in range(Tm):
        mask = masks[t] if masks is not None and masks.shape[0] > t else None
        flow_feats[t] = safe_median_masked(flow_mags[t], mask)
    last = float(flow_feats[-1]) if Tm > 0 else 0.0
    flow_feats = np.concatenate([flow_feats, np.array([last], dtype=np.float32)])
    return flow_feats

def compute_pose_velocity(keypoints):
    """Compute mean L2 displacement across available joints for each frame.
       Accepts (T, J, 2/3) or (T, F) flattened triplet layout."""
    if keypoints is None:
        return None
    if keypoints.ndim == 3 and keypoints.shape[2] >= 2:
        coords = keypoints[..., :2]
        T = coords.shape[0]
        vel = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            prev = coords[t-1]
            cur  = coords[t]
            valid = ~np.isnan(prev[:,0]) & ~np.isnan(cur[:,0])
            if valid.sum() == 0:
                vel[t] = 0.0
            else:
                d = cur[valid] - prev[valid]
                vel[t] = float(np.mean(np.linalg.norm(d, axis=1)))
        return vel
    elif keypoints.ndim == 2:
        # flattened triplet layout (T, F) where F % 3 == 0
        T, F = keypoints.shape
        if F % 3 != 0:
            # fallback: compute diff on full vector
            dd = np.linalg.norm(np.diff(keypoints, axis=0), axis=1)
            return np.concatenate([np.zeros(1, dtype=np.float32), dd]).astype(np.float32)
        trip = keypoints.reshape(T, -1, 3)
        coords = trip[..., :2]
        vel = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            prev = coords[t-1]
            cur  = coords[t]
            valid = ~np.isnan(prev[:,0]) & ~np.isnan(cur[:,0])
            if valid.sum() == 0:
                vel[t] = 0.0
            else:
                d = cur[valid] - prev[valid]
                vel[t] = float(np.mean(np.linalg.norm(d, axis=1)))
        return vel
    else:
        return None

def compute_hand_angle_mean(keypoints, hand_offset_col=132):
    """Compute a stable per-frame average hand motion direction (angle).
       Accepts flattened (T, F) or (T, J, 3). Returns length T (first elem 0)."""
    if keypoints is None:
        return None
    if keypoints.ndim == 3:
        trip = keypoints.reshape(keypoints.shape[0], -1, keypoints.shape[2])
        flat = trip.reshape(keypoints.shape[0], -1)
    else:
        flat = keypoints
    # guard
    if flat.shape[1] <= hand_offset_col:
        # fallback: zeros
        return np.zeros(flat.shape[0], dtype=np.float32)
    hand_flat = flat[:, hand_offset_col:]
    # reshaping into triplets
    if hand_flat.shape[1] % 3 != 0:
        return np.zeros(flat.shape[0], dtype=np.float32)
    hand_trip = hand_flat.reshape(hand_flat.shape[0], -1, 3)
    T = hand_trip.shape[0]
    angles = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        prev = hand_trip[t-1,:,:2]
        cur  = hand_trip[t,:,:2]
        valid = ~np.isnan(prev[:,0]) & ~np.isnan(cur[:,0])
        if valid.sum() == 0:
            angles[t] = 0.0
            continue
        vecs = cur[valid] - prev[valid]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        avg_unit = (vecs / norms).mean(axis=0)
        angles[t] = float(math.atan2(float(avg_unit[1]), float(avg_unit[0])))
    return angles

def compute_frame_diffs(frames):
    """Compute mean absolute difference per-frame between consecutive frames.
       frames: (T,H,W,3) uint8 or float in [0,255]"""
    if frames is None:
        return None
    frames_f = frames.astype(np.float32) / 255.0
    diffs = np.mean(np.abs(frames_f[1:] - frames_f[:-1]), axis=(1,2,3))
    diffs_padded = np.concatenate([diffs, diffs[-1:]]) if diffs.size>0 else np.zeros((frames.shape[0],), dtype=np.float32)
    return diffs_padded.astype(np.float32)

def zscore_fill(features):
    feats = np.array(features, dtype=np.float32)
    # nan -> mean (per-col)
    col_mean = np.nanmean(feats, axis=0)
    inds = np.where(np.isnan(feats))
    feats[inds] = np.take(col_mean, inds[1])
    std = feats.std(axis=0) + 1e-6
    feats = (feats - feats.mean(axis=0)) / std
    return feats

def safe_kmeans_medoid_selection(features, num_keyframes, overcluster=2, random_state=0):
    n = features.shape[0]
    if n <= 0:
        return []
    k = min(max(1, num_keyframes + overcluster), n)
    if n <= k:
        # fallback: spread indices uniformly
        return list(np.linspace(0, n-1, min(num_keyframes, n), dtype=int))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(features)
    medoids = []
    for c in range(k):
        cluster_indices = np.where(kmeans.labels_ == c)[0]
        if cluster_indices.size == 0:
            continue
        center = kmeans.cluster_centers_[c]
        dists = np.linalg.norm(features[cluster_indices] - center, axis=1)
        medoid_idx = int(cluster_indices[np.argmin(dists)])
        medoids.append(medoid_idx)
    return medoids

def temporal_greedy_selection(candidates, scores, num_select, min_dist=3):
    if len(candidates) == 0:
        return []
    cand = sorted(np.unique(candidates))
    # sort by score desc
    cand_sorted = sorted(cand, key=lambda i: -float(scores[i]))
    selected = []
    for idx in cand_sorted:
        if len(selected) >= num_select:
            break
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)
    if len(selected) < num_select:
        remaining = [i for i in cand if i not in selected]
        rem_sorted = sorted(remaining, key=lambda i: -float(scores[i]))
        for r in rem_sorted:
            if len(selected) >= num_select: break
            selected.append(r)
    return sorted(selected)

def compute_per_frame_features(frames, keypoints, flow_mags, masks):
    """Return features (T, F) and helper arrays used later."""
    T = frames.shape[0]
    flow_feats = compute_flow_feature_per_frame(flow_mags, masks) if flow_mags is not None else np.zeros(T, dtype=np.float32)
    pose_vel = compute_pose_velocity(keypoints) if keypoints is not None else np.zeros(T, dtype=np.float32)
    angles = compute_hand_angle_mean(keypoints) if keypoints is not None else np.zeros(T, dtype=np.float32)
    frame_diffs = compute_frame_diffs(frames) if frames is not None else np.zeros(T, dtype=np.float32)
    # smoothing scalars a little
    flow_feats_s = gaussian_filter1d(flow_feats, sigma=1.2) if flow_feats is not None else flow_feats
    pose_vel_s = gaussian_filter1d(pose_vel, sigma=1.2) if pose_vel is not None else pose_vel
    angles_s = gaussian_filter1d(angles, sigma=1.2) if angles is not None else angles
    frame_diffs_s = gaussian_filter1d(frame_diffs, sigma=1.0) if frame_diffs is not None else frame_diffs
    features = np.stack([flow_feats_s, pose_vel_s, angles_s, frame_diffs_s], axis=1)
    features = zscore_fill(features)
    return features, dict(flow=flow_feats_s, pose_vel=pose_vel_s, angles=angles_s, frame_diffs=frame_diffs_s)

# ---------------------------
# Main selection pipeline
# ---------------------------

def select_keyframes(video_dir, out_dir=None, num_keyframes=8, pca_dims=4, min_dist=4, hand_offset_col=132):
    os.makedirs(out_dir if out_dir is not None else os.path.join(video_dir, "keyframes"), exist_ok=True)
    out_dir = out_dir or os.path.join(video_dir, "keyframes")
    # load
    frames = load_frames(video_dir)
    keypoints = load_optional_npy(os.path.join(video_dir, "keypoints.npy"))
    flow_mags = load_optional_npy(os.path.join(video_dir, "flow_magnitudes.npy"))
    masks = load_optional_npy(os.path.join(video_dir, "masks.npy"))
    attn = load_optional_npy(os.path.join(video_dir, "attention_mask.npy"))
    T = frames.shape[0]
    if attn is not None:
        real_len = int(np.sum(attn))
        # clip things
        T = real_len
        frames = frames[:real_len]
        if keypoints is not None:
            keypoints = keypoints[:real_len]
        if masks is not None:
            masks = masks[:real_len]
        if flow_mags is not None and flow_mags.shape[0] >= real_len-1:
            flow_mags = flow_mags[:real_len-1]
    else:
        real_len = T

    # compute features
    features, perfs = compute_per_frame_features(frames, keypoints, flow_mags, masks)

    # PCA reduction if possible
    n_samples, n_feats = features.shape
    comp = min(pca_dims, n_feats, n_samples)
    if comp >= 1 and comp < n_feats:
        pca = PCA(n_components=comp, random_state=0)
        feats_red = pca.fit_transform(features)
    else:
        feats_red = features

    # medoid selection overclustered
    medoids = safe_kmeans_medoid_selection(feats_red, num_keyframes, overcluster=2)

    # fused scores
    fused = np.nanmean(features, axis=1)
    # dynamic threshold for peaks
    thr = fused.mean() + 0.5 * fused.std()
    peaks, props = find_peaks(fused, height=thr, distance=2)
    if peaks.size == 0:
        # fallback: pick top-K fused frames
        fallback = np.argsort(fused)[-min(3, n_samples):]
        peaks = fallback

    # union candidates
    candidates = np.unique(np.concatenate([medoids, peaks, [0, real_len-1]])).astype(int)

    # temporal greedy selection to enforce diversity
    selected = temporal_greedy_selection(candidates, fused, num_keyframes, min_dist=min_dist)

    # confidence filtering (compute per-frame mean conf if possible)
    conf_mean = None
    if keypoints is not None:
        if keypoints.ndim == 3 and keypoints.shape[2] >= 3:
            confs = keypoints[..., 2].mean(axis=1)
            conf_mean = confs
        elif keypoints.ndim == 2:
            T_k, F = keypoints.shape
            if F % 3 == 0:
                trip = keypoints.reshape(T_k, -1, 3)
                conf_mean = trip[:,:,2].mean(axis=1)
    if conf_mean is not None:
        # relax threshold
        thr_conf = max(0.35, float(conf_mean.mean() - 0.1))
        kept = [i for i in selected if conf_mean[i] >= thr_conf]
        if len(kept) == 0:
            # fall back: keep top conf frames among selected
            kept = sorted(selected, key=lambda i: -conf_mean[i])[:min(len(selected), num_keyframes)]
        selected = sorted(kept)

    # final safety: if not enough selected, fill from top fused frames ensuring distance
    if len(selected) < num_keyframes:
        pool = np.argsort(fused)[::-1]
        for idx in pool:
            if idx in selected: continue
            if all(abs(idx - s) >= min_dist for s in selected):
                selected.append(int(idx))
            if len(selected) >= num_keyframes:
                break
    selected = sorted(list(np.unique(selected)))[:num_keyframes]

    # Save outputs
    key_indices = np.array(selected, dtype=int)
    key_frames = frames[key_indices]
    key_kps = keypoints[key_indices] if keypoints is not None else None
    np.save(os.path.join(out_dir, "key_indices.npy"), key_indices)
    np.save(os.path.join(out_dir, "key_frames.npy"), key_frames)
    if key_kps is not None:
        np.save(os.path.join(out_dir, "key_keypoints.npy"), key_kps)

    # write PNGs and a preview grid
    for i, idx in enumerate(key_indices):
        outp = os.path.join(out_dir, f"keyframe_{idx:04d}.png")
        im = key_frames[i]
        # ensure RGB->BGR for cv2.imwrite
        cv2.imwrite(outp, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    # preview grid (concatenate horizontally)
    cols = min(len(key_indices), 8)
    thumb_h = 160
    thumbs = []
    for im in key_frames[:cols]:
        # resize preserving aspect
        h, w = im.shape[:2]
        scale = thumb_h / h
        imr = cv2.resize(im, (int(w*scale), thumb_h))
        thumbs.append(imr)
    if len(thumbs) > 0:
        grid = np.hstack(thumbs)
        cv2.imwrite(os.path.join(out_dir, "preview_selected.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # QC JSON
    qc = {
        "video_dir": video_dir,
        "real_len": int(real_len),
        "selected_indices": key_indices.tolist(),
        "median_flow_mean": float(np.nanmedian(perfs["flow"])),
        "pose_vel_mean": float(np.nanmean(perfs["pose_vel"])),
        "frame_diff_mean": float(np.nanmean(perfs["frame_diffs"])),
        "conf_mean_mean": float(conf_mean.mean()) if conf_mean is not None else None
    }
    with open(os.path.join(out_dir, "qc.json"), "w") as f:
        json.dump(qc, f, indent=2)

    print(f"[done] selected {len(key_indices)} keyframes -> {out_dir}")
    return key_indices, out_dir

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from a preprocessed video folder.")
    parser.add_argument("--video_dir", type=str, required=True, help="path to preprocessed/<video_id>/")
    parser.add_argument("--num_keyframes", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--pca_dims", type=int, default=4)
    parser.add_argument("--min_dist", type=int, default=4)
    args = parser.parse_args()

    # quick check for missing common files
    required = ["frames.npy", "keypoints.npy"]
    missing = [r for r in required if not os.path.exists(os.path.join(args.video_dir, r))]
    if missing:
        print("Warning: some expected files are missing from the folder:", missing)
        print("This script supports frames.npy OR frames_resized/; please provide one of them.")
    select_keyframes(args.video_dir, out_dir=args.out_dir, num_keyframes=args.num_keyframes, pca_dims=args.pca_dims, min_dist=args.min_dist)

if __name__ == "__main__":
    main()