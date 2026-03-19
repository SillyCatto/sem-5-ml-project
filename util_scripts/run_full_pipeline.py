#!/usr/bin/env python3
"""
Run the full sign-language pipeline from raw videos to model training.

Stages:
1) Preprocess raw videos
2) Extract keyframes (batch)
3) Extract landmarks (batch)
4) Train model

Example:
  .\\.venv\\Scripts\\python.exe util_scripts\\run_full_pipeline.py \
      --dataset-dir dataset/raw_video_data \
      --preprocessed-dir outputs/preprocessed \
      --keyframes-dir outputs/keyframes \
      --landmarks-dir outputs/landmarks \
      --checkpoint-dir checkpoints/rtx2060_full_run \
      --trainer-script "model/trainer_current_config(gpu).py" \
      --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing import PipelineConfig as PreprocessConfig
from app.preprocessing import PreprocessingPipeline
from app.core.keyframe_pipeline import PipelineConfig as KeyframeConfig
from app.core.keyframe_pipeline import run_batch as run_keyframe_batch
from app.core.landmark_pipeline import LandmarkConfig
from app.core.landmark_pipeline import run_batch as run_landmark_batch


def _load_train_model(trainer_script: Path):
    """Load train_model function from an arbitrary trainer .py file path."""
    trainer_script = trainer_script.resolve()
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

    spec = importlib.util.spec_from_file_location("pipeline_trainer", str(trainer_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module from: {trainer_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "train_model"):
        raise AttributeError(
            f"train_model() not found in trainer script: {trainer_script}"
        )

    return module.train_model


def _count_class_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return len([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing -> keyframe -> landmark -> training in one CLI command"
    )

    # Inputs/outputs
    parser.add_argument("--dataset-dir", default="dataset/raw_video_data")
    parser.add_argument("--preprocessed-dir", default="outputs/preprocessed")
    parser.add_argument("--keyframes-dir", default="outputs/keyframes")
    parser.add_argument("--landmarks-dir", default="outputs/landmarks")
    parser.add_argument("--checkpoint-dir", default="checkpoints/full_pipeline_run")
    parser.add_argument(
        "--trainer-script",
        default="model/trainer_current_config(gpu).py",
        help="Path to a trainer script that defines train_model(...)",
    )

    # Stage toggles
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-keyframes", action="store_true")
    parser.add_argument("--skip-landmarks", action="store_true")
    parser.add_argument("--skip-training", action="store_true")

    # Preprocessing options
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--no-skip-existing", action="store_true")

    # Keyframe options
    parser.add_argument("--min-keyframes", type=int, default=8)
    parser.add_argument("--max-keyframes", type=int, default=15)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--hold-threshold", type=float, default=0.12)

    # Landmark options
    parser.add_argument("--target-seq-len", type=int, default=15)
    parser.add_argument("--scale-factor", type=float, default=100.0)

    # Training options
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "transformer", "hybrid"])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--empty-cache-steps", type=int, default=0)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    preprocessed_dir = (PROJECT_ROOT / args.preprocessed_dir).resolve()
    keyframes_dir = (PROJECT_ROOT / args.keyframes_dir).resolve()
    landmarks_dir = (PROJECT_ROOT / args.landmarks_dir).resolve()
    checkpoint_dir = (PROJECT_ROOT / args.checkpoint_dir).resolve()
    trainer_script = (PROJECT_ROOT / args.trainer_script).resolve()

    print("=" * 72)
    print("FULL PIPELINE RUNNER")
    print("=" * 72)
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Dataset dir      : {dataset_dir}")
    print(f"Preprocessed dir : {preprocessed_dir}")
    print(f"Keyframes dir    : {keyframes_dir}")
    print(f"Landmarks dir    : {landmarks_dir}")
    print(f"Checkpoint dir   : {checkpoint_dir}")
    print(f"Trainer script   : {trainer_script}")

    # 1) Preprocessing
    if not args.skip_preprocessing:
        print("\n[1/4] Preprocessing videos...")
        pp_cfg = PreprocessConfig(
            dataset_dir=dataset_dir,
            output_dir=preprocessed_dir,
            target_fps=args.target_fps,
            output_size=args.output_size,
            clahe_clip_limit=args.clahe_clip,
        )
        pipeline = PreprocessingPipeline(pp_cfg)

        summary = pipeline.run(skip_existing=not args.no_skip_existing)
        print(
            "Preprocessing summary: "
            f"total={summary['total']} processed={summary['processed']} "
            f"skipped={summary['skipped']} failed={summary['failed']}"
        )
        if summary["failed"] > 0:
            print("[warn] Some preprocessing jobs failed. Downstream stages may process fewer samples.")
    else:
        print("\n[1/4] Skipped preprocessing.")

    # 2) Keyframes
    if not args.skip_keyframes:
        print("\n[2/4] Extracting keyframes...")
        kf_cfg = KeyframeConfig(
            min_keyframes=args.min_keyframes,
            max_keyframes=args.max_keyframes,
            smooth_sigma=args.smooth_sigma,
            sandwiched_hold_threshold=args.hold_threshold,
        )
        kf_results = run_keyframe_batch(
            dataset_dir=str(preprocessed_dir),
            output_root=str(keyframes_dir),
            config=kf_cfg,
        )
        ok = sum(1 for r in kf_results if r.get("status") == "✓")
        fail = len(kf_results) - ok
        print(f"Keyframe summary: total={len(kf_results)} success={ok} failed={fail}")
        if fail > 0:
            print("[warn] Some keyframe jobs failed. Landmark/training set will be smaller.")
    else:
        print("\n[2/4] Skipped keyframe extraction.")

    # 3) Landmarks
    if not args.skip_landmarks:
        print("\n[3/4] Extracting landmarks...")
        lm_cfg = LandmarkConfig(
            target_sequence_length=args.target_seq_len,
            scale_factor=args.scale_factor,
        )
        lm_results = run_landmark_batch(
            root_dir=str(keyframes_dir),
            output_root=str(landmarks_dir),
            config=lm_cfg,
        )
        ok = sum(1 for r in lm_results if r.get("status") == "✓")
        fail = len(lm_results) - ok
        print(f"Landmark summary: total={len(lm_results)} success={ok} failed={fail}")
        if fail > 0:
            print("[warn] Some landmark jobs failed. Training will use available samples only.")
    else:
        print("\n[3/4] Skipped landmark extraction.")

    # 4) Training
    if not args.skip_training:
        print("\n[4/4] Training model...")
        train_model = _load_train_model(trainer_script)

        inferred_classes = _count_class_dirs(landmarks_dir)
        num_classes = args.num_classes if args.num_classes is not None else inferred_classes
        if num_classes <= 0:
            raise RuntimeError(
                "No class directories found in landmarks output. "
                "Run landmark extraction first or pass --num-classes explicitly."
            )

        train_model(
            landmarks_dir=str(landmarks_dir),
            flow_dir=None,
            model_type=args.model_type,
            num_classes=num_classes,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_dir=str(checkpoint_dir),
            device=args.device,
            num_workers=args.num_workers,
            use_mixed_precision=args.use_mixed_precision,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            empty_cache_steps=args.empty_cache_steps,
        )

        print("\nTraining completed.")
        print(f"Checkpoints/logs/history saved under: {checkpoint_dir}")
        print(f"Best model path: {checkpoint_dir / 'best_model.pth'}")
        print(f"TensorBoard log dir: {checkpoint_dir / 'logs'}")
        print(f"Training history: {checkpoint_dir / 'training_history.json'}")
    else:
        print("\n[4/4] Skipped training.")

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
