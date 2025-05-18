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
"""

import os
import sys
import yaml
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import our project modules
from dataset_loader import DatasetLoader
from model import SSLPretrainer, Classifier, CombinedModel
from trainer import Trainer
from evaluation import Evaluation

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
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    
    # Set defaults for training config
    training_config = config.get("training", {})
    training_config["learning_rate"] = training_config.get("learning_rate", DEFAULT_LEARNING_RATE)
    training_config["batch_size"] = training_config.get("batch_size", DEFAULT_BATCH_SIZE)
    training_config["pretraining_epochs"] = training_config.get("pretraining_epochs", DEFAULT_PRETRAIN_EPOCHS)
    training_config["supervised_epochs"] = training_config.get("supervised_epochs", DEFAULT_SUPERVISED_EPOCHS)
    training_config["mc_dropout_passes"] = training_config.get("mc_dropout_passes", DEFAULT_MC_DROPOUT_PASSES)
    config["training"] = training_config

    # Set defaults for model config, particularly for encoder, decoder and classifier
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder", {})
    encoder_config["patch_size"] = encoder_config.get("patch_size", 16)
    encoder_config["num_layers"] = encoder_config.get("num_layers", 6)
    encoder_config["hidden_dim"] = encoder_config.get("hidden_dim", 128)
    encoder_config["architecture"] = encoder_config.get("architecture", "Vision Transformer")
    model_config["encoder"] = encoder_config

    decoder_config = model_config.get("decoder", {})
    decoder_config["num_layers"] = decoder_config.get("num_layers", 2)
    decoder_config["hidden_dim"] = decoder_config.get("hidden_dim", encoder_config["hidden_dim"])
    decoder_config["architecture"] = decoder_config.get("architecture", "Transformer")
    model_config["decoder"] = decoder_config
    
    classifier_config = model_config.get("classifier", {})
    classifier_config["activation"] = classifier_config.get("activation", "Parametric ReLU")
    classifier_config["dropout"] = classifier_config.get("dropout", 0.1)
    model_config["classifier"] = classifier_config
    config["model"] = model_config

    # Set defaults for data config
    data_config = config.get("data", {})
    data_config["num_channels"] = data_config.get("num_channels", DEFAULT_NUM_CHANNELS)
    data_config["patch_window_size"] = data_config.get("patch_window_size", DEFAULT_PATCH_WINDOW_SIZE)
    data_config["mask_ratio"] = data_config.get("mask_ratio", DEFAULT_MASK_RATIO)
    data_config["undersample_filter_ratio"] = data_config.get("undersample_filter_ratio", DEFAULT_UNDERSAMPLE_FILTER_RATIO)
    
    # Set default file paths if not provided
    data_config["explanatory_raster_path"] = data_config.get("explanatory_raster_path", "data/explanatory.tif")
    data_config["label_raster_paths"] = data_config.get("label_raster_paths", "data/label.tif")
    config["data"] = data_config

    # Set random seed
    config["random_seed"] = config.get("random_seed", DEFAULT_RANDOM_SEED)

    logger.info("Configuration loaded and defaults set: %s", config)
    return config

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seeds set to %d", seed)

def main():
    # 1. Load configuration and initialize environment.
    config = load_config("config.yaml")
    seed = config.get("random_seed", DEFAULT_RANDOM_SEED)
    set_random_seeds(seed)

    # 2. Instantiate the DatasetLoader and load data.
    dataset_loader = DatasetLoader(config)
    try:
        X_raw, Y_raw = dataset_loader.load_geospatial_data()
    except Exception as e:
        logger.error("Failed to load geospatial data: %s", e)
        sys.exit(1)
    
    # Preprocess the explanatory raster
    X_processed = dataset_loader.preprocess_data(X_raw)
    patch_window_size = config["data"]["patch_window_size"]
    patches = dataset_loader.create_patches(X_processed, patch_window_size)
    labels = dataset_loader.extract_labels(Y_raw, patch_window_size)

    # Convert patches and labels to numpy arrays then to torch tensors.
    # patches is a list of np.ndarray with shape (num_channels, w, w)
    try:
        samples_np = np.array(patches)  # shape: (num_samples, num_channels, w, w)
    except Exception as e:
        logger.error("Error converting patches to numpy array: %s", e)
        sys.exit(1)
    samples_tensor = torch.tensor(samples_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Split data into train, validation, and test sets (80/10/10)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset_loader.split_data(patches, labels, seed=seed)
    # Convert split data to tensors
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
    X_val_tensor = torch.tensor(np.array(X_val), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.long)
    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)

    # Create DataLoader for SSL pretraining with all patches (using samples_tensor)
    ssl_dataset = DataLoader(samples_tensor, batch_size=config["training"]["batch_size"], shuffle=True)

    # 3. Initialize Models:
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

    # 4. Set up the Trainer and train:
    # Trainer requires training and validation data for supervised fine-tuning.
    trainer = Trainer(
        combined_model,
        train_data=(X_train_tensor, y_train_tensor),
        val_data=(X_val_tensor, y_val_tensor),
        config=config
    )
    
    # Train the SSL module
    logger.info("Starting SSL pretraining...")
    trainer.train_ssl_pretrainer(ssl_dataset)
    
    # Train the supervised classifier (with undersampling strategy)
    logger.info("Starting supervised fine-tuning...")
    trainer.train_classifier()
    
    # 5. Evaluate the model on the test set:
    # Create Evaluation instance with test data tuple (X_test_tensor, y_test_tensor)
    evaluation = Evaluation(model=combined_model, test_data=(X_test_tensor, y_test_tensor), config=config)
    metrics = evaluation.evaluate_metrics()
    logger.info("Final Evaluation Metrics: %s", metrics)
    
    # Optional: Run ablation study with 50% feature dropout to assess robustness.
    ablation_metrics = evaluation.run_ablation(drop_ratio=0.5)
    logger.info("Ablation Study Metrics (50%% dropout): %s", ablation_metrics)
    
    # Optional: Explain a sample prediction using Integrated Gradients.
    # Select one sample from test set.
    sample = X_test_tensor[0].unsqueeze(0)  # shape: [1, channels, w, w]
    # Use a zero-tensor baseline
    baseline = torch.zeros_like(sample)
    attributions = evaluation.explain_sample(sample, baseline=baseline, steps=50)
    logger.info("Integrated Gradients Attribution for one sample computed with shape: %s", attributions.shape)
    
    # Optionally, save prospectivity maps and uncertainty outputs if required.
    # For demonstration, perform MC dropout inference on test set:
    combined_model.eval()
    with torch.no_grad():
        mean_preds, uncertainty = combined_model.predict(X_test_tensor.to(trainer.device),
                                                           mc_passes=config["training"]["mc_dropout_passes"])
    logger.info("MC Dropout Inference completed on test set. Prediction mean shape: %s, Uncertainty shape: %s",
                mean_preds.shape, uncertainty.shape)
    
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
