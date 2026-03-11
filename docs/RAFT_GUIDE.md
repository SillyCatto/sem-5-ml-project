# RAFT Optical Flow - Complete Guide

## ✅ What's Now Available in Streamlit

Your Streamlit app now has **4 tabs** with full RAFT integration:

### 1. 📹 Keyframe Extractor
Extract keyframes from sign language videos

### 2. 🦴 Landmark Extractor  
Extract MediaPipe landmarks from keyframes

### 3. 🌊 **NEW: Flow Extractor**
**Extract RAFT optical flow features from videos**

Features:
- ✅ Single video processing (upload or select from directory)
- ✅ Batch processing (process entire class directories)
- ✅ Pre-trained RAFT models (small/large)
- ✅ MPS/CUDA/CPU support
- ✅ Configurable feature dimensions
- ✅ Saves `.npy` files compatible with training

### 4. 🤖 Model Training

**Train Tab:**
- ✅ Train with landmarks only
- ✅ Train with landmarks + flow features
- ✅ Three model architectures (LSTM/Transformer/Hybrid)
- ✅ MPS acceleration on M4 MacBook
- ✅ Real-time training metrics

**Test Tab:**
- Coming soon (placeholder)

**Inference Tab:**
- ✅ **Upload a video** → Auto-extract landmarks + flow → Predict
- ✅ **Upload landmarks (.npy)** → Predict
- ✅ Optional flow extraction for videos
- ✅ Top-K predictions with confidence scores
- ✅ Works with trained models

---

## 🚀 Complete Workflow: Video → Prediction

### Option A: End-to-End Video Inference (Easiest)

1. **Train a model** (Tab 4 - Train):
   ```
   - Set Landmarks Directory: extracted_landmarks
   - Optional Flow Directory: extracted_flow (or leave empty)
   - Configure model settings
   - Click "Start Training"
   ```

2. **Test on new video** (Tab 4 - Inference):
   ```
   - Upload a sign language video (.mp4, .avi, etc.)
   - Check "Extract Flow Features" if your model uses flow
   - Select RAFT model (small/large)
   - Click "Predict"
   ```
   
   **The app will automatically:**
   - Extract MediaPipe landmarks from video
   - Extract RAFT optical flow (if enabled)
   - Run inference with your trained model
   - Show top-K predictions with confidence

### Option B: Pre-Extract Flow for Training

1. **Extract flow features** (Tab 3 - Flow Extractor):
   
   **Single Video:**
   ```
   - Upload or select video
   - Choose RAFT model (small/large)
   - Set feature dimension (default 32)
   - Set output directory
   - Click "Extract Flow Features"
   ```
   
   **Batch Processing:**
   ```
   Input structure:
   videos/
   ├── hello/
   │   ├── sample1.mp4
   │   └── sample2.mp4
   ├── yes/
   │   └── sample1.mp4
   
   Output will be:
   extracted_flow/
   ├── hello/
   │   ├── sample1.npy
   │   └── sample2.npy
   ├── yes/
   │   └── sample1.npy
   ```

2. **Train with flow features** (Tab 4 - Train):
   ```
   - Landmarks Directory: extracted_landmarks
   - Flow Directory: extracted_flow
   - The dataset loader will automatically match files by name
   ```

---

## 💡 Quick Test

### Test RAFT Extraction Right Now:

1. Start Streamlit:
   ```bash
   streamlit run keyframe_extractor/app.py
   ```

2. Go to **Tab 3: 🌊 Flow Extractor**

3. Upload any video or use test video

4. Click "🚀 Extract Flow Features"

5. See flow features saved as `.npy` file

### Test Inference Right Now:

1. Go to **Tab 4: 🤖 Model Training**

2. Click **"Inference"** sub-tab

3. Upload a sign language video

4. Check "Extract Flow Features"

5. Click "🎯 Predict"

6. See top predictions with confidence scores!

---

## 📊 What Gets Extracted

### Landmarks (258 features per frame)
- Face landmarks: 468 × 3 → subset
- Pose landmarks: 33 × 3 → subset  
- Hand landmarks: 21 × 2 hands × 3 → subset
- **Total: 258 features per frame**

### RAFT Flow (32 features per frame)
- Optical flow between consecutive frames
- Flow field: (H, W, 2) per frame pair
- Global pooling: mean, max, std
- **Total: 32 features per frame** (configurable)

### Combined Features (290 total)
- Landmarks: 258 dimensions
- Flow: 32 dimensions
- Concatenated for hybrid models

---

## 🎯 Model Architectures

All models support **landmarks only** OR **landmarks + flow**:

### LSTM
- Bidirectional LSTM
- Fast training and inference
- Good for sequential patterns
- Hidden dim: 256 (default)

### Transformer
- Self-attention mechanism
- Better long-range dependencies
- Slower but more accurate
- d_model: 256, heads: 8

### Hybrid
- LSTM for landmarks
- Transformer for flow
- Best accuracy
- Combines both strengths

---

## ⚙️ Performance Tips

### For M4 MacBook:
- ✅ Use `device="mps"` for GPU acceleration
- ✅ RAFT small model: ~2-3 sec/video
- ✅ RAFT large model: ~5-8 sec/video
- ✅ Mixed precision training: enabled by default

### Memory Optimization:
- Use batch_size=16 for training
- RAFT small model uses less VRAM
- Process videos in batches for large datasets

### Speed vs Accuracy:
- **Fastest**: Landmarks only + LSTM
- **Balanced**: Landmarks + flow (small) + LSTM  
- **Most Accurate**: Landmarks + flow (large) + Hybrid

---

## 📁 File Organization

```
project/
├── videos/                      # Original videos
│   ├── class1/*.mp4
│   └── class2/*.mp4
├── keyframes/                   # Extracted keyframes
│   ├── class1/*.jpg
│   └── class2/*.jpg
├── extracted_landmarks/         # MediaPipe landmarks
│   ├── class1/*.npy            # (N, 258)
│   └── class2/*.npy
├── extracted_flow/              # RAFT flow features
│   ├── class1/*.npy            # (N, 32)
│   └── class2/*.npy
└── checkpoints/                 # Trained models
    └── lstm_run1/
        ├── best_model.pth
        └── training_history.json
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'raft_flow_extractor'"
- Make sure you're in the project directory
- The flow_page.py adds `model/` to sys.path automatically

### "Flow extraction is slow"
- Use RAFT small model instead of large
- Check device is set to "mps" (not "cpu")
- Reduce video resolution if very high

### "Prediction fails with flow features"
- Make sure your model was trained WITH flow features
- Check flow directory was provided during training
- Verify flow feature dimension matches (default 32)

---

## ✨ Summary

**YES - RAFT is fully integrated in Streamlit!**

✅ Extract flow from videos (single or batch)  
✅ Train models with landmarks + flow  
✅ Test directly on videos with automatic feature extraction  
✅ Pre-trained RAFT models (no training needed)  
✅ MPS acceleration on M4 MacBook  
✅ Complete end-to-end workflow in UI

You can now **upload any sign language video** and get predictions with RAFT optical flow **directly from the Streamlit interface**! 🎉
