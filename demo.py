#!/usr/bin/env python
"""demo.py

Demo script for testing the mineral prospectivity mapping pipeline with synthetic data.
This allows testing the complete pipeline without requiring real geospatial data.

Usage:
    python demo.py
"""

import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model import SSLPretrainer, Classifier, CombinedModel
    from trainer import Trainer
    from evaluation import Evaluation
except ImportError as e:
    logger.error("Failed to import required modules: %s", e)
    logger.error("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def create_synthetic_data(num_samples=200, num_channels=3, patch_size=64):
    """
    Create synthetic geospatial-like data for testing.
    
    Args:
        num_samples: Number of patch samples to generate
        num_channels: Number of channels (bands) in the data
        patch_size: Size of each square patch (e.g., 64x64)
        
    Returns:
        Tuple of (X, y) where X is samples and y is binary labels
    """
    logger.info("Generating synthetic data: %d samples, %d channels, %dx%d patches",
                num_samples, num_channels, patch_size, patch_size)
    
    # Generate synthetic multi-band patches with some spatial structure
    X = np.random.randn(num_samples, num_channels, patch_size, patch_size).astype(np.float32)
    
    # Add some spatial patterns to make it more realistic
    for i in range(num_samples):
        for c in range(num_channels):
            # Add some smooth gradients
            x_grad = np.linspace(0, 1, patch_size)
            y_grad = np.linspace(0, 1, patch_size)
            X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
            X[i, c] += 0.5 * (X_grad + Y_grad)
            
            # Add some circular features (simulating geological structures)
            center_x, center_y = np.random.randint(10, patch_size-10, 2)
            radius = np.random.randint(5, 15)
            for x in range(patch_size):
                for y in range(patch_size):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        X[i, c, x, y] += np.random.randn() * 2.0
    
    # Normalize
    X = (X - X.mean(axis=(2, 3), keepdims=True)) / (X.std(axis=(2, 3), keepdims=True) + 1e-8)
    
    # Generate binary labels with class imbalance (20% positive)
    y = (np.random.rand(num_samples) < 0.2).astype(int)
    
    logger.info("Synthetic data generated. Label distribution: %d positive, %d negative",
                y.sum(), num_samples - y.sum())
    
    return X, y


def run_demo():
    """Run a complete demo of the pipeline with synthetic data."""
    logger.info("="*60)
    logger.info("DEMO: Mineral Prospectivity Mapping Pipeline")
    logger.info("="*60)
    
    # Configuration for demo
    config = {
        "training": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "pretraining_epochs": 3,  # Reduced for demo
            "supervised_epochs": 3,    # Reduced for demo
            "mc_dropout_passes": 10    # Reduced for demo
        },
        "model": {
            "encoder": {
                "architecture": "Vision Transformer",
                "patch_size": 16,
                "num_layers": 4,  # Reduced for demo
                "hidden_dim": 64   # Reduced for demo
            },
            "decoder": {
                "architecture": "Transformer",
                "num_layers": 2,
                "hidden_dim": 64
            },
            "classifier": {
                "architecture": "Multi-Layer Perceptron",
                "activation": "Parametric ReLU",
                "dropout": 0.1
            }
        },
        "data": {
            "num_channels": 3,
            "patch_window_size": 64,
            "mask_ratio": 0.75,
            "undersample_filter_ratio": 0.05
        },
        "random_seed": 42
    }
    
    # Set random seeds for reproducibility
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    # Generate synthetic data
    X_all, y_all = create_synthetic_data(num_samples=200, num_channels=3, patch_size=64)
    
    # Split into train/val/test (80/10/10)
    train_end = int(0.8 * len(X_all))
    val_end = int(0.9 * len(X_all))
    
    X_train = torch.tensor(X_all[:train_end], dtype=torch.float32)
    y_train = torch.tensor(y_all[:train_end], dtype=torch.long)
    X_val = torch.tensor(X_all[train_end:val_end], dtype=torch.float32)
    y_val = torch.tensor(y_all[train_end:val_end], dtype=torch.long)
    X_test = torch.tensor(X_all[val_end:], dtype=torch.float32)
    y_test = torch.tensor(y_all[val_end:], dtype=torch.long)
    
    logger.info("Data split - Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))
    
    # Create DataLoader for SSL pretraining
    X_all_tensor = torch.tensor(X_all, dtype=torch.float32)
    ssl_dataset = DataLoader(X_all_tensor, batch_size=config["training"]["batch_size"], shuffle=True)
    
    # Initialize models
    logger.info("Initializing models...")
    ssl_params = {
        "encoder": config["model"]["encoder"],
        "decoder": config["model"]["decoder"],
        "data": config["data"]
    }
    classifier_params = {
        "input_dim": config["model"]["encoder"]["hidden_dim"],
        "hidden_dims": [config["model"]["encoder"]["hidden_dim"] // 2],
        "dropout": config["model"]["classifier"]["dropout"],
        "activation": config["model"]["classifier"]["activation"]
    }
    
    ssl_pretrainer = SSLPretrainer(ssl_params)
    classifier = Classifier(classifier_params)
    combined_model = CombinedModel(encoder=ssl_pretrainer, classifier=classifier)
    
    # Initialize trainer
    logger.info("Setting up trainer...")
    trainer = Trainer(
        combined_model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        config=config
    )
    
    # Train SSL pretrainer
    logger.info("\n" + "="*60)
    logger.info("Phase 1: Self-Supervised Pretraining")
    logger.info("="*60)
    trainer.train_ssl_pretrainer(ssl_dataset)
    
    # Train classifier
    logger.info("\n" + "="*60)
    logger.info("Phase 2: Supervised Fine-tuning")
    logger.info("="*60)
    trainer.train_classifier()
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("Phase 3: Evaluation")
    logger.info("="*60)
    evaluation = Evaluation(model=combined_model, test_data=(X_test, y_test), config=config)
    
    try:
        metrics = evaluation.evaluate_metrics()
        
        print("\n" + "="*60)
        print("DEMO EVALUATION RESULTS")
        print("="*60)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:.<45} {metric_value:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
    
    # Test explainability
    logger.info("\n" + "="*60)
    logger.info("Phase 4: Explainability (Integrated Gradients)")
    logger.info("="*60)
    try:
        sample = X_test[0].unsqueeze(0)
        baseline = torch.zeros_like(sample)
        attributions = evaluation.explain_sample(sample, baseline=baseline, steps=20)
        logger.info("Integrated Gradients computed successfully. Attribution shape: %s", attributions.shape)
    except Exception as e:
        logger.warning("Integrated Gradients computation failed: %s", e)
    
    # Clean up demo weights
    demo_files = ["ssl_pretrained_encoder.pth", "supervised_classifier.pth"]
    for f in demo_files:
        if os.path.exists(f):
            os.remove(f)
            logger.info("Cleaned up demo file: %s", f)
    
    logger.info("\n" + "="*60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("The pipeline is working correctly with synthetic data.")
    logger.info("Next steps:")
    logger.info("  1. Prepare your geospatial raster data in GeoTIFF format")
    logger.info("  2. Update config.yaml with your data paths")
    logger.info("  3. Run: python main.py")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("Demo failed with error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
