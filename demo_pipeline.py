#!/usr/bin/env python3
"""
Demo: RAFT + Landmarks Sign Language Recognition Pipeline

This script demonstrates the complete pipeline:
1. Extract optical flow using RAFT
2. Load landmarks from MediaPipe
3. Train a sign language classifier
4. Evaluate the model

Run: python demo_pipeline.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add model directory to path
sys.path.append(str(Path(__file__).parent))

from model.raft_flow_extractor import RAFTFlowExtractor
from model.dataset import SignLanguageDataset
from model.sign_classifier import create_model


def check_environment():
    """Check if environment is properly configured."""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check PyTorch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # Check MPS
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) available")
        device = "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA available")
        device = "cuda"
    else:
        print("✓ Using CPU")
        device = "cpu"
    
    print(f"✓ Selected device: {device}")
    
    return device


def demo_raft_extraction():
    """Demo 1: Extract optical flow using RAFT."""
    print("\n" + "=" * 70)
    print("DEMO 1: RAFT OPTICAL FLOW EXTRACTION")
    print("=" * 70)
    
    print("\nInitializing RAFT model...")
    extractor = RAFTFlowExtractor(model_size="small")
    
    print("\nCreating sample video frames...")
    # Create 5 frames with a moving object
    frames = []
    for i in range(5):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Moving circle
        import cv2
        cv2.circle(frame, (50 + i * 30, 120), 20, (255, 100, 100), -1)
        frames.append(frame)
    
    print(f"Created {len(frames)} frames of size {frames[0].shape}")
    
    print("\nExtracting optical flow...")
    flows = extractor.extract_flow_from_frames(frames, return_magnitude=True)
    print(f"✓ Flow shape: {flows.shape}")
    print(f"  - {flows.shape[0]} flow fields (N-1 frames)")
    print(f"  - Size: {flows.shape[1]}x{flows.shape[2]}")
    print(f"  - Channels: {flows.shape[3]} (magnitude + angle)")
    
    # Extract global features
    print("\nExtracting global flow features...")
    global_flow = extractor.extract_global_flow_features(flows, pool_size=(4, 4))
    print(f"✓ Global flow features: {global_flow.shape}")
    print(f"  - {global_flow.shape[0]} temporal steps")
    print(f"  - {global_flow.shape[1]} feature dimensions")
    
    return flows


def demo_dataset_loading():
    """Demo 2: Load sign language dataset."""
    print("\n" + "=" * 70)
    print("DEMO 2: DATASET LOADING")
    print("=" * 70)
    
    # Check if landmarks exist
    landmarks_dir = Path("extracted_landmarks")
    
    if not landmarks_dir.exists():
        print(f"\n⚠️  Landmarks directory not found: {landmarks_dir}")
        print("   To use this demo:")
        print("   1. Run the Streamlit app: streamlit run keyframe_extractor/app.py")
        print("   2. Use the Landmark Extractor tab to process videos")
        print("   3. Re-run this demo")
        return None
    
    print(f"\nLoading dataset from: {landmarks_dir}")
    
    try:
        dataset = SignLanguageDataset(landmarks_dir=landmarks_dir)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of classes: {len(dataset.classes)}")
        print(f"  - Classes: {', '.join(dataset.classes)}")
        
        # Show distribution
        print("\n  Class distribution:")
        dist = dataset.get_class_distribution()
        for cls, count in dist.items():
            print(f"    {cls}: {count} samples")
        
        # Load a sample
        if len(dataset) > 0:
            features, label = dataset[0]
            print(f"\n  Sample 0:")
            print(f"    - Landmarks shape: {features['landmarks'].shape}")
            print(f"    - Flow shape: {features['flow'].shape}")
            print(f"    - Label: {label} ({dataset.classes[label]})")
        
        return dataset
    
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        return None


def demo_model_creation(device):
    """Demo 3: Create and test models."""
    print("\n" + "=" * 70)
    print("DEMO 3: MODEL ARCHITECTURES")
    print("=" * 70)
    
    # Create dummy data
    batch_size = 4
    num_frames = 30
    landmark_dim = 258
    flow_dim = 32
    num_classes = 15
    
    print(f"\nCreating test batch:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Frames: {num_frames}")
    print(f"  - Landmark features: {landmark_dim}")
    print(f"  - Flow features: {flow_dim}")
    
    landmarks = torch.randn(batch_size, num_frames, landmark_dim).to(device)
    flow = torch.randn(batch_size, num_frames - 1, flow_dim).to(device)
    
    models_to_test = ["lstm", "transformer", "hybrid"]
    
    for model_type in models_to_test:
        print(f"\n--- {model_type.upper()} Model ---")
        
        try:
            model = create_model(
                model_type=model_type,
                num_classes=num_classes,
                device=device
            )
            
            # Forward pass
            with torch.no_grad():
                logits = model(landmarks, flow)
            
            print(f"✓ Forward pass successful")
            print(f"  - Input: landmarks {landmarks.shape}, flow {flow.shape}")
            print(f"  - Output: {logits.shape}")
            print(f"  - Predictions: {torch.argmax(logits, dim=1).tolist()}")
            
            # Calculate parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"  - Parameters: {params:,}")
            
        except Exception as e:
            print(f"❌ Failed to create {model_type} model: {e}")


def demo_training_setup():
    """Demo 4: Training setup (without actually training)."""
    print("\n" + "=" * 70)
    print("DEMO 4: TRAINING SETUP")
    print("=" * 70)
    
    print("\nTo train a model:")
    print("\nOption 1: Use the Streamlit UI")
    print("  1. Run: streamlit run keyframe_extractor/app.py")
    print("  2. Go to the '🤖 Model Training' tab")
    print("  3. Configure training parameters")
    print("  4. Click 'Start Training'")
    
    print("\nOption 2: Use Python script")
    print("  Example code:")
    print("""
    from model.trainer import train_model
    
    train_model(
        landmarks_dir="extracted_landmarks",
        flow_dir=None,  # Optional
        model_type="lstm",
        num_classes=15,
        batch_size=16,
        num_epochs=50,
        learning_rate=0.001,
        save_dir="checkpoints/my_model"
    )
    """)
    
    print("\nOption 3: Use the demo training script")
    print("  python model/trainer.py")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("SIGN LANGUAGE RECOGNITION PIPELINE DEMO")
    print("RAFT Optical Flow + MediaPipe Landmarks + LSTM/Transformer")
    print("=" * 70)
    
    # Check environment
    device = check_environment()
    
    # Demo 1: RAFT extraction
    demo_raft_extraction()
    
    # Demo 2: Dataset loading
    dataset = demo_dataset_loading()
    
    # Demo 3: Model creation
    demo_model_creation(device)
    
    # Demo 4: Training setup
    demo_training_setup()
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Extract landmarks from your videos using the Streamlit app")
    print("2. (Optional) Extract optical flow features")
    print("3. Train a model using the Streamlit UI or Python script")
    print("4. Evaluate and test your trained model")
    
    print("\nFor questions or issues:")
    print("- Check SETUP_COMPLETE.md for detailed documentation")
    print("- Run verify_setup.py to check your environment")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
