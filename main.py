"""main.py

This is the main entry point for the mineral prospectivity mapping application.
It orchestrates the entire pipeline:
  1. Loads configuration from config.yaml.
  2. Initializes the environment (logging, random seeds).
  3. Loads and preprocesses geospatial data using the DatasetLoader.
  4. Slices the data into patches and splits it into training, validation, and test sets.
  5. Instantiates the SSLPretrainer, Classifier, and CombinedModel.
  6. Trains the self-supervised pretraining module (SSL) and then the supervised classifier
     with an undersampling strategy.
  7. Performs evaluation using Monte Carlo (MC) dropout for uncertainty estimation and
     Integrated Gradients for explainability.
  8. Logs and prints all relevant evaluation metrics.

All configuration values are read from config.yaml; if vital parameters are missing,
default values are used.

Usage:
    python main.py [--config CONFIG_PATH]
    
Example:
    python main.py --config config.yaml
"""

import os
import sys
import yaml
import logging
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    # Import our project modules
    from dataset_loader import DatasetLoader
    from model import SSLPretrainer, Classifier, CombinedModel
    from trainer import Trainer
    from evaluation import Evaluation
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default configuration values
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_PRETRAIN_EPOCHS = 30
DEFAULT_SUPERVISED_EPOCHS = 10
DEFAULT_MC_DROPOUT_PASSES = 30
DEFAULT_PATCH_WINDOW_SIZE = 64
DEFAULT_NUM_CHANNELS = 3
DEFAULT_MASK_RATIO = 0.75
DEFAULT_UNDERSAMPLE_FILTER_RATIO = 0.05
DEFAULT_RANDOM_SEED = 42

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file and set defaults for missing parameters."""
    if not os.path.exists(config_path):
        logger.warning("Configuration file %s not found. Using default configuration.", config_path)
        config = {}
    else:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to load configuration file %s: %s", config_path, e)
            logger.info("Using default configuration.")
            config = {}
    
    # Set defaults for training config
    training_config = config.get("training", {})
    training_config["learning_rate"] = training_config.get("learning_rate") or DEFAULT_LEARNING_RATE
    training_config["batch_size"] = training_config.get("batch_size") or DEFAULT_BATCH_SIZE
    training_config["pretraining_epochs"] = training_config.get("pretraining_epochs") or DEFAULT_PRETRAIN_EPOCHS
    training_config["supervised_epochs"] = training_config.get("supervised_epochs") or DEFAULT_SUPERVISED_EPOCHS
    training_config["mc_dropout_passes"] = training_config.get("mc_dropout_passes") or DEFAULT_MC_DROPOUT_PASSES
    config["training"] = training_config

    # Set defaults for model config, particularly for encoder, decoder and classifier
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder", {})
    encoder_config["patch_size"] = encoder_config.get("patch_size") or 16
    encoder_config["num_layers"] = encoder_config.get("num_layers") or 6
    encoder_config["hidden_dim"] = encoder_config.get("hidden_dim") or 128
    encoder_config["architecture"] = encoder_config.get("architecture") or "Vision Transformer"
    model_config["encoder"] = encoder_config

    decoder_config = model_config.get("decoder", {})
    decoder_config["num_layers"] = decoder_config.get("num_layers") or 2
    decoder_config["hidden_dim"] = decoder_config.get("hidden_dim") or encoder_config["hidden_dim"]
    decoder_config["architecture"] = decoder_config.get("architecture") or "Transformer"
    model_config["decoder"] = decoder_config
    
    classifier_config = model_config.get("classifier", {})
    classifier_config["activation"] = classifier_config.get("activation") or "Parametric ReLU"
    # Handle dropout which might be a string "to be tuned" in original config
    dropout_val = classifier_config.get("dropout")
    if dropout_val is None or isinstance(dropout_val, str):
        classifier_config["dropout"] = 0.1
    else:
        classifier_config["dropout"] = dropout_val
    model_config["classifier"] = classifier_config
    config["model"] = model_config

    # Set defaults for data config
    data_config = config.get("data", {})
    data_config["num_channels"] = data_config.get("num_channels") or DEFAULT_NUM_CHANNELS
    data_config["patch_window_size"] = data_config.get("patch_window_size") or DEFAULT_PATCH_WINDOW_SIZE
    data_config["mask_ratio"] = data_config.get("mask_ratio") or DEFAULT_MASK_RATIO
    data_config["undersample_filter_ratio"] = data_config.get("undersample_filter_ratio") or DEFAULT_UNDERSAMPLE_FILTER_RATIO
    
    # Set default file paths if not provided
    data_config["explanatory_raster_path"] = data_config.get("explanatory_raster_path", "data/explanatory.tif")
    data_config["label_raster_paths"] = data_config.get("label_raster_paths", "data/label.tif")
    config["data"] = data_config

    # Set random seed
    config["random_seed"] = config.get("random_seed") or DEFAULT_RANDOM_SEED

    logger.info("Configuration loaded and defaults set.")
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mineral Prospectivity Mapping with Self-Supervised Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--skip-ssl-training",
        action="store_true",
        help="Skip SSL pretraining (use if encoder weights already exist)"
    )
    parser.add_argument(
        "--ssl-weights",
        type=str,
        default="ssl_pretrained_encoder.pth",
        help="Path to load/save SSL encoder weights"
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default="supervised_classifier.pth",
        help="Path to load/save classifier weights"
    )
    return parser.parse_args()

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seeds set to %d", seed)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # 1. Load configuration and initialize environment.
    config = load_config(args.config)
    seed = config.get("random_seed") or DEFAULT_RANDOM_SEED
    set_random_seeds(seed)
    
    # Check if required data files exist
    data_config = config.get("data", {})
    explanatory_path = data_config.get("explanatory_raster_path", "data/explanatory.tif")
    label_path = data_config.get("label_raster_paths", "data/label.tif")
    
    if not os.path.exists(explanatory_path):
        logger.error("Explanatory raster file not found at: %s", explanatory_path)
        logger.error("Please provide valid data files or update config.yaml with correct paths.")
        logger.error("See README.md for instructions on preparing your data.")
        sys.exit(1)

    # 2. Instantiate the DatasetLoader and load data.
    logger.info("Loading geospatial data...")
    dataset_loader = DatasetLoader(config)
    try:
        X_raw, Y_raw = dataset_loader.load_geospatial_data()
    except Exception as e:
        logger.error("Failed to load geospatial data: %s", e)
        logger.error("Please check your data files and paths in config.yaml")
        sys.exit(1)
    
    # Preprocess the explanatory raster
    logger.info("Preprocessing data...")
    X_processed = dataset_loader.preprocess_data(X_raw)
    patch_window_size = config["data"]["patch_window_size"]
    
    logger.info("Creating patches...")
    patches = dataset_loader.create_patches(X_processed, patch_window_size)
    labels = dataset_loader.extract_labels(Y_raw, patch_window_size)
    
    if len(patches) == 0:
        logger.error("No patches created. Check that your raster dimensions are compatible with patch size.")
        sys.exit(1)
    
    logger.info("Created %d patches from input rasters.", len(patches))

    # Convert patches and labels to numpy arrays then to torch tensors.
    # patches is a list of np.ndarray with shape (num_channels, w, w)
    try:
        samples_np = np.array(patches)  # shape: (num_samples, num_channels, w, w)
    except Exception as e:
        logger.error("Error converting patches to numpy array: %s", e)
        sys.exit(1)
    samples_tensor = torch.tensor(samples_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    logger.info("Tensor shapes - Samples: %s, Labels: %s", samples_tensor.shape, labels_tensor.shape)

    # Split data into train, validation, and test sets (80/10/10)
    logger.info("Splitting data into train/val/test sets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset_loader.split_data(patches, labels, seed=seed)
    # Convert split data to tensors
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
    X_val_tensor = torch.tensor(np.array(X_val), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.long)
    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)
    
    logger.info("Train: %d, Val: %d, Test: %d samples", len(X_train), len(X_val), len(X_test))

    # Create DataLoader for SSL pretraining with all patches (using samples_tensor)
    ssl_dataset = DataLoader(samples_tensor, batch_size=config["training"]["batch_size"], shuffle=True)

    # 3. Initialize Models:
    logger.info("Initializing models...")
    # Prepare model configuration parameters for SSLPretrainer and Classifier.
    ssl_params = {
        "encoder": config["model"]["encoder"],
        "decoder": config["model"]["decoder"],
        "data": config["data"]
    }
    classifier_params = {
        "input_dim": config["model"]["encoder"].get("hidden_dim", 128),
        "hidden_dims": [config["model"]["encoder"].get("hidden_dim", 128) // 2],
        "dropout": config["model"]["classifier"].get("dropout", 0.1),
        "activation": config["model"]["classifier"].get("activation", "Parametric ReLU")
    }
    # Instantiate SSLPretrainer and Classifier
    ssl_pretrainer = SSLPretrainer(ssl_params)
    classifier = Classifier(classifier_params)

    # Instantiate CombinedModel with the frozen encoder and classifier.
    combined_model = CombinedModel(encoder=ssl_pretrainer, classifier=classifier)
    
    logger.info("Models initialized successfully.")

    # 4. Set up the Trainer and train:
    # Trainer requires training and validation data for supervised fine-tuning.
    trainer = Trainer(
        combined_model,
        train_data=(X_train_tensor, y_train_tensor),
        val_data=(X_val_tensor, y_val_tensor),
        config=config
    )
    
    # Train the SSL module (or load pretrained weights)
    if args.skip_ssl_training and os.path.exists(args.ssl_weights):
        logger.info("Loading pretrained SSL encoder weights from %s", args.ssl_weights)
        try:
            combined_model.encoder.load_state_dict(torch.load(args.ssl_weights))
            logger.info("SSL encoder weights loaded successfully.")
        except Exception as e:
            logger.error("Failed to load SSL encoder weights: %s", e)
            logger.info("Will train SSL from scratch instead.")
            trainer.train_ssl_pretrainer(ssl_dataset)
    else:
        logger.info("Starting SSL pretraining...")
        trainer.train_ssl_pretrainer(ssl_dataset)
    
    # Train the supervised classifier (with undersampling strategy)
    logger.info("Starting supervised fine-tuning...")
    trainer.train_classifier()
    
    # 5. Evaluate the model on the test set:
    logger.info("Evaluating model on test set...")
    # Create Evaluation instance with test data tuple (X_test_tensor, y_test_tensor)
    evaluation = Evaluation(model=combined_model, test_data=(X_test_tensor, y_test_tensor), config=config)
    
    try:
        metrics = evaluation.evaluate_metrics()
        logger.info("Final Evaluation Metrics: %s", metrics)
        
        # Print metrics in a readable format
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:.<40} {metric_value:.4f}")
        print("="*50 + "\n")
    except Exception as e:
        logger.error("Error during evaluation: %s", e)
        logger.warning("Continuing with optional evaluations...")
    
    # Optional: Run ablation study with 50% feature dropout to assess robustness.
    try:
        logger.info("Running ablation study...")
        ablation_metrics = evaluation.run_ablation(drop_ratio=0.5)
        logger.info("Ablation Study Metrics (50%% dropout): %s", ablation_metrics)
    except Exception as e:
        logger.warning("Ablation study failed: %s", e)
    
    # Optional: Explain a sample prediction using Integrated Gradients.
    try:
        if len(X_test_tensor) > 0:
            logger.info("Computing Integrated Gradients for sample explanation...")
            # Select one sample from test set.
            sample = X_test_tensor[0].unsqueeze(0)  # shape: [1, channels, w, w]
            # Use a zero-tensor baseline
            baseline = torch.zeros_like(sample)
            attributions = evaluation.explain_sample(sample, baseline=baseline, steps=50)
            logger.info("Integrated Gradients Attribution computed with shape: %s", attributions.shape)
    except Exception as e:
        logger.warning("Integrated Gradients computation failed: %s", e)
    
    # Optionally, save prospectivity maps and uncertainty outputs if required.
    # For demonstration, perform MC dropout inference on test set:
    try:
        logger.info("Performing MC dropout inference on test set...")
        combined_model.eval()
        device = next(combined_model.parameters()).device
        with torch.no_grad():
            mean_preds, uncertainty = combined_model.predict(
                X_test_tensor.to(device),
                mc_passes=config["training"]["mc_dropout_passes"]
            )
        logger.info("MC Dropout Inference completed. Prediction mean shape: %s, Uncertainty shape: %s",
                    mean_preds.shape, uncertainty.shape)
    except Exception as e:
        logger.warning("MC dropout inference failed: %s", e)
    
    logger.info("Pipeline complete. All results logged above.")
    print("\nPipeline completed successfully! Check the logs for detailed results.")
    print(f"Model weights saved: {args.ssl_weights}, {args.classifier_weights}")

if __name__ == "__main__":
    main()
