"""trainer.py

This module implements the Trainer class that orchestrates both the self-supervised
pretraining of the SSLPretrainer (encoder-decoder) and the supervised fine-tuning of the
Classifier using the frozen encoder from the SSLPretrainer.
It also implements undersampling of unknown samples based on Euclidean similarity,
and integrates monitoring metrics such as reconstruction loss, SSIM, PSNR (for SSL pretraining),
and BCE along with standard classification metrics for supervised training.
"""

import os
import math
import logging
import random
from typing import Any, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# Import models and configuration defaults are expected from config.yaml via config dict
# We assume that the CombinedModel (which includes the frozen SSLPretrainer and the Classifier)
# has been instantiated outside and passed to the Trainer.

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration settings (if not provided by config.yaml)
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_PRETRAIN_EPOCHS = 30
DEFAULT_SUPERVISED_EPOCHS = 10
DEFAULT_MC_DROPOUT_PASSES = 30
DEFAULT_UNDERSAMPLE_FILTER_RATIO = 0.05  # 5% by default

class Trainer:
    """
    Trainer class manages two phases of training:
      1. Self-Supervised Pretraining:
         Trains the SSLPretrainer on unlabeled patches with a masked image modeling objective.
      2. Supervised Fine-Tuning:
         Trains the Classifier using features extracted by the frozen encoder.
         Uses an undersampling strategy based on Euclidean distances between unknown and positive samples.
         
    Methods:
        train_ssl_pretrainer(ssl_dataset: DataLoader, val_dataset: DataLoader) -> None:
            Trains the SSLPretrainer for a configured number of epochs.
        train_classifier(
            train_samples: torch.Tensor,
            train_labels: torch.Tensor,
            val_samples: torch.Tensor,
            val_labels: torch.Tensor
        ) -> None:
            Trains the classifier using the undersampling strategy and BCE loss.
    """

    def __init__(self, 
                 combined_model: Any, 
                 train_data: Any, 
                 val_data: Any, 
                 config: Dict) -> None:
        """
        Initialize the Trainer with the combined model (SSLPretrainer + Classifier),
        training/validation data and configuration parameters.
        
        Args:
            combined_model (CombinedModel): Combined model including the frozen encoder and the classifier.
            train_data (Any): Training dataset (for supervised fine-tuning). This is expected to be a tuple (X_train, y_train) as tensors.
            val_data (Any): Validation dataset (for supervised fine-tuning). Tuple (X_val, y_val).
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.combined_model = combined_model

        # Unpack training data for supervised fine-tuning
        self.X_train_supervised, self.y_train_supervised = train_data
        self.X_val_supervised, self.y_val_supervised = val_data

        # Configuration for training hyperparameters
        training_config = config.get("training", {})
        model_config = config.get("model", {})
        data_config = config.get("data", {})

        # Set hyperparameters with default values if None.
        self.learning_rate = training_config.get("learning_rate") if training_config.get("learning_rate") is not None else DEFAULT_LEARNING_RATE
        self.batch_size = training_config.get("batch_size") if training_config.get("batch_size") is not None else DEFAULT_BATCH_SIZE
        self.pretraining_epochs = training_config.get("pretraining_epochs") if training_config.get("pretraining_epochs") is not None else DEFAULT_PRETRAIN_EPOCHS
        self.supervised_epochs = training_config.get("supervised_epochs") if training_config.get("supervised_epochs") is not None else DEFAULT_SUPERVISED_EPOCHS
        self.mc_dropout_passes = training_config.get("mc_dropout_passes") if training_config.get("mc_dropout_passes") is not None else DEFAULT_MC_DROPOUT_PASSES
        self.undersample_filter_ratio = data_config.get("undersample_filter_ratio") if data_config.get("undersample_filter_ratio") is not None else DEFAULT_UNDERSAMPLE_FILTER_RATIO

        # Optimizer for SSL pretraining: update all parameters of the SSLPretrainer (encoder-decoder).
        # We assume that the combined_model.encoder is the SSLPretrainer.
        self.ssl_optimizer = optim.Adam(self.combined_model.encoder.parameters(), lr=self.learning_rate)

        # Optimizer for supervised fine-tuning: only update parameters of the classifier.
        self.classifier_optimizer = optim.Adam(self.combined_model.classifier.parameters(), lr=self.learning_rate)

        # Loss functions
        self.mse_loss_fn = nn.MSELoss()
        self.bce_loss_fn = nn.BCELoss()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.combined_model.to(self.device)

        # Set random seed for reproducibility
        seed = config.get("random_seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        logger.info("Trainer initialized with learning_rate=%.6f, batch_size=%d, pretraining_epochs=%d, supervised_epochs=%d, mc_dropout_passes=%d",
                    self.learning_rate, self.batch_size, self.pretraining_epochs, self.supervised_epochs, self.mc_dropout_passes)

    def _compute_ssim_psnr(self, target: np.ndarray, reconstruction: np.ndarray) -> Tuple[float, float]:
        """
        Compute SSIM and PSNR between target and reconstructed images.
        
        Args:
            target (np.ndarray): Ground truth image. (H, W) or (C, H, W) - for simplicity assume single channel by averaging if needed.
            reconstruction (np.ndarray): Reconstructed image.
            
        Returns:
            Tuple[float, float]: (ssim, psnr)
        """
        # If multi-channel, convert to grayscale by averaging over channels.
        if target.ndim == 3 and target.shape[0] > 1:
            target_gray = np.mean(target, axis=0)
            reconstruction_gray = np.mean(reconstruction, axis=0)
        else:
            target_gray = target.squeeze()
            reconstruction_gray = reconstruction.squeeze()

        # Compute SSIM and PSNR using skimage metrics.
        ssim_val = structural_similarity(target_gray, reconstruction_gray, data_range=target_gray.max() - target_gray.min())
        psnr_val = peak_signal_noise_ratio(target_gray, reconstruction_gray, data_range=target_gray.max() - target_gray.min())
        return ssim_val, psnr_val

    def train_ssl_pretrainer(self, ssl_dataset: DataLoader, val_dataset: DataLoader = None) -> None:
        """
        Train the SSLPretrainer (encoder-decoder) on the unlabeled dataset using the masked image modeling task.
        
        Args:
            ssl_dataset (DataLoader): DataLoader for self-supervised pretraining data (unlabeled patches).
            val_dataset (DataLoader, optional): DataLoader for validation dataset for reconstruction metrics.
        """
        logger.info("Starting SSL pretraining for %d epochs.", self.pretraining_epochs)
        self.combined_model.encoder.train()  # Set encoder to training mode
        for epoch in range(1, self.pretraining_epochs + 1):
            epoch_loss = 0.0
            ssim_list = []
            psnr_list = []
            batch_count = 0

            for batch in ssl_dataset:
                # Assume batch is a tensor of shape [batch, channels, height, width]
                x = batch.to(self.device)
                self.ssl_optimizer.zero_grad()
                # Forward pass: get reconstruction from SSLPretrainer (encoder-decoder)
                reconstruction = self.combined_model.encoder(x)
                loss = self.mse_loss_fn(reconstruction, x)
                loss.backward()
                self.ssl_optimizer.step()

                epoch_loss += loss.item()
                # Compute metrics SSIM and PSNR on CPU for logging
                x_cpu = x.detach().cpu().numpy()
                recon_cpu = reconstruction.detach().cpu().numpy()
                for i in range(x_cpu.shape[0]):
                    ssim_val, psnr_val = self._compute_ssim_psnr(x_cpu[i], recon_cpu[i])
                    ssim_list.append(ssim_val)
                    psnr_list.append(psnr_val)
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
            avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
            logger.info("Epoch [%d/%d] - SSL Loss: %.6f, SSIM: %.4f, PSNR: %.4f", 
                        epoch, self.pretraining_epochs, avg_loss, avg_ssim, avg_psnr)

            # Optional: validate on val_dataset if provided.
            if val_dataset is not None:
                self.combined_model.encoder.eval()
                val_loss = 0.0
                val_ssim_list = []
                val_psnr_list = []
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataset:
                        x_val = val_batch.to(self.device)
                        reconstruction_val = self.combined_model.encoder(x_val)
                        loss_val = self.mse_loss_fn(reconstruction_val, x_val)
                        val_loss += loss_val.item()
                        x_val_cpu = x_val.detach().cpu().numpy()
                        recon_val_cpu = reconstruction_val.detach().cpu().numpy()
                        for i in range(x_val_cpu.shape[0]):
                            ssim_val, psnr_val = self._compute_ssim_psnr(x_val_cpu[i], recon_val_cpu[i])
                            val_ssim_list.append(ssim_val)
                            val_psnr_list.append(psnr_val)
                        val_batches += 1

                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                avg_val_ssim = np.mean(val_ssim_list) if val_ssim_list else 0.0
                avg_val_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
                logger.info("Validation - SSL Loss: %.6f, SSIM: %.4f, PSNR: %.4f", 
                            avg_val_loss, avg_val_ssim, avg_val_psnr)
                self.combined_model.encoder.train()

        # After SSL pretraining, save encoder weights.
        encoder_weights_path = "ssl_pretrained_encoder.pth"
        torch.save(self.combined_model.encoder.state_dict(), encoder_weights_path)
        logger.info("SSL pretraining complete. Encoder weights saved to %s", encoder_weights_path)

    def _undersample_supervised_data(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply undersampling strategy on supervised training data:
          - Extract features using the frozen encoder.
          - Separate positive samples (label == 1) and unknown/negative samples.
          - For each unknown sample, compute the minimum Euclidean distance to any positive sample.
          - Filter out the top fraction (as per undersample_filter_ratio) of unknown samples that are most similar to positive samples.
          - Optionally balance classes by oversampling the minority class.
        
        Args:
            X (torch.Tensor): Tensor of input samples [N, channels, H, W].
            y (torch.Tensor): Tensor of labels [N]. Expected binary labels with 1 (positive) and 0 (unknown/absent).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Filtered X and corresponding y.
        """
        self.combined_model.encoder.eval()
        with torch.no_grad():
            features = self.combined_model.extract_features(X.to(self.device))  # [N, feature_dim]
        features_np = features.cpu().numpy()
        y_np = y.cpu().numpy()

        # Identify indices for positive and unknown (assumed 0 for unknown/negative) samples.
        pos_indices = np.where(y_np == 1)[0]
        unknown_indices = np.where(y_np == 0)[0]

        if len(pos_indices) == 0 or len(unknown_indices) == 0:
            logger.warning("No positive or unknown samples found for undersampling. Returning original data.")
            return X, y

        pos_features = features_np[pos_indices]  # shape: [P, feature_dim]

        # Compute minimal Euclidean distance for each unknown sample to any positive sample.
        distances = []
        for idx in unknown_indices:
            unknown_feat = features_np[idx]
            # Compute Euclidean distances to all positive features
            dists = np.linalg.norm(pos_features - unknown_feat, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)
        distances = np.array(distances)
        
        # Determine threshold based on undersample_filter_ratio.
        num_to_filter = int(len(distances) * self.undersample_filter_ratio)
        if num_to_filter < 1:
            num_to_filter = 1
        # Get indices of unknown samples to filter (i.e., smallest distances)
        filter_order = np.argsort(distances)  # smallest distances first
        filter_unknown_indices = unknown_indices[filter_order[:num_to_filter]]
        
        # Create mask for supervised data: mark all positive samples as True, and only keep unknown samples not in filter_unknown_indices.
        keep_indices = list(pos_indices) + [idx for idx in unknown_indices if idx not in filter_unknown_indices]
        X_filtered = X[keep_indices]
        y_filtered = y[keep_indices]

        logger.info("Undersampling applied: filtered out %d out of %d unknown samples. Final supervised samples: %d",
                    len(filter_unknown_indices), len(unknown_indices), X_filtered.size(0))

        # Optional: Oversample minority class if imbalance exists.
        # Count class frequencies
        unique, counts = np.unique(y_filtered.cpu().numpy(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        if 1 in class_counts and 0 in class_counts:
            pos_count = class_counts[1]
            neg_count = class_counts[0]
            if pos_count < neg_count:
                # Oversample positive samples to match number of negatives
                pos_indices_filtered = np.where(y_filtered.cpu().numpy() == 1)[0]
                factor = int(np.ceil(neg_count / pos_count))
                X_pos = X_filtered[pos_indices_filtered]
                y_pos = y_filtered[pos_indices_filtered]
                X_oversampled = X_pos.repeat(factor, dim=0)
                y_oversampled = y_pos.repeat(factor, dim=0)
                # Combine original negatives and oversampled positives
                neg_indices_filtered = np.where(y_filtered.cpu().numpy() == 0)[0]
                X_neg = X_filtered[neg_indices_filtered]
                y_neg = y_filtered[neg_indices_filtered]
                X_final = torch.cat([X_neg, X_oversampled], dim=0)
                y_final = torch.cat([y_neg, y_oversampled], dim=0)
                logger.info("Oversampling applied: Positives oversampled from %d to %d to balance negatives %d.", 
                            pos_count, X_oversampled.size(0), neg_count)
                return X_final, y_final
        # If not imbalanced, return filtered data without oversampling.
        return X_filtered, y_filtered

    def train_classifier(self) -> None:
        """
        Train the supervised classifier using features extracted by the frozen encoder.
        Applies undersampling to mitigate labeling unknown true positives as negatives.
        Uses Binary Cross Entropy loss for training.
        """
        logger.info("Starting supervised fine-tuning for %d epochs.", self.supervised_epochs)
        # Convert supervised training data to tensors if not already done.
        X_train_tensor = self.X_train_supervised.to(self.device)
        y_train_tensor = self.y_train_supervised.to(self.device).float().unsqueeze(1)  # shape: [N, 1]

        # Apply undersampling on the training data.
        X_train_filtered, y_train_filtered = self._undersample_supervised_data(X_train_tensor, y_train_tensor)

        # Create DataLoader for supervised training
        supervised_dataset = TensorDataset(X_train_filtered, y_train_filtered)
        train_loader = DataLoader(supervised_dataset, batch_size=self.batch_size, shuffle=True)

        # Set classifier to training mode; encoder is frozen.
        self.combined_model.classifier.train()
        for epoch in range(1, self.supervised_epochs + 1):
            epoch_loss = 0.0
            batch_count = 0
            for batch in train_loader:
                x_batch, y_batch = batch
                self.classifier_optimizer.zero_grad()
                # Forward pass: the combined model's forward extracts features via frozen encoder and then passes through classifier.
                predictions = self.combined_model(x_batch)
                loss = self.bce_loss_fn(predictions, y_batch)
                loss.backward()
                self.classifier_optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            logger.info("Supervised Epoch [%d/%d] - BCE Loss: %.6f", epoch, self.supervised_epochs, avg_loss)

            # Optionally, evaluate on validation set after each epoch.
            self.evaluate_classifier(validation=True)

        # After training, enable MC dropout (by keeping classifier in train mode) for inference.
        self.combined_model.classifier.train()
        # Save classifier checkpoint.
        classifier_weights_path = "supervised_classifier.pth"
        torch.save(self.combined_model.classifier.state_dict(), classifier_weights_path)
        logger.info("Supervised fine-tuning complete. Classifier weights saved to %s", classifier_weights_path)

    def evaluate_classifier(self, validation: bool = True) -> None:
        """
        Evaluate the supervised classifier on the validation set using BCE loss and log basic metrics.
        For simplicity, this implementation logs only the BCE loss.
        
        Args:
            validation (bool): If True, evaluates on the validation dataset.
        """
        self.combined_model.eval()
        if validation:
            X_val = self.X_val_supervised.to(self.device)
            y_val = self.y_val_supervised.to(self.device).float().unsqueeze(1)
            with torch.no_grad():
                predictions = self.combined_model(X_val)
                loss = self.bce_loss_fn(predictions, y_val)
            logger.info("Validation BCE Loss: %.6f", loss.item())
        self.combined_model.train()

# Example usage of Trainer class (this code would be executed by the main application)
if __name__ == "__main__":
    # Dummy instantiation for testing purposes.
    # In practice, datasets would be created and CombinedModel instantiated properly.

    # Import necessary modules from dataset_loader and model.
    from dataset_loader import DatasetLoader
    from model import SSLPretrainer, Classifier, CombinedModel

    # Example configuration dictionary (would normally be loaded from config.yaml)
    config = {
        "training": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "pretraining_epochs": 30,
            "supervised_epochs": 10,
            "mc_dropout_passes": 30
        },
        "model": {
            "encoder": {
                "architecture": "Vision Transformer",
                "patch_size": 16,
                "num_layers": 6,
                "hidden_dim": 128
            },
            "decoder": {
                "architecture": "Transformer",
                "num_layers": 2,
                "hidden_dim": 128
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
            "undersample_filter_ratio": 0.05,
            "explanatory_raster_path": "data/explanatory.tif",  # example paths
            "label_raster_paths": "data/label.tif"
        },
        "random_seed": 42
    }

    # Instantiate DatasetLoader and load dummy data (here we simulate with random tensors)
    # In practice, you would load real geospatial data.
    dummy_num_samples = 100
    patch_window_size = config["data"]["patch_window_size"]
    num_channels = config["data"]["num_channels"]
    
    # Create random tensors as dummy data for SSL pretraining and supervised training.
    ssl_data = torch.randn(dummy_num_samples, num_channels, patch_window_size, patch_window_size)
    supervised_labels = (torch.rand(dummy_num_samples) > 0.5).long()  # binary labels 0 or 1

    # Split the dummy supervised data into train and validation sets.
    split_idx = int(0.8 * dummy_num_samples)
    X_train_sup = ssl_data[:split_idx]
    y_train_sup = supervised_labels[:split_idx]
    X_val_sup = ssl_data[split_idx:]
    y_val_sup = supervised_labels[split_idx:]

    # Create DataLoader for SSL pretraining
    ssl_dataset = DataLoader(ssl_data, batch_size=config["training"]["batch_size"], shuffle=True)

    # Instantiate models: SSLPretrainer and Classifier
    ssl_params = {
        "encoder": config["model"]["encoder"],
        "decoder": config["model"]["decoder"],
        "data": config["data"]
    }
    classifier_params = {
        "input_dim": config["model"]["encoder"].get("hidden_dim", 128),
        "hidden_dims": [64],
        "dropout": config["model"]["classifier"].get("dropout", 0.1),
        "activation": config["model"]["classifier"].get("activation", "Parametric ReLU")
    }
    ssl_pretrainer = SSLPretrainer(ssl_params)
    classifier = Classifier(classifier_params)

    # CombinedModel: integrates frozen encoder and classifier.
    combined_model = CombinedModel(encoder=ssl_pretrainer, classifier=classifier)

    # Instantiate the Trainer with supervised training data and configuration.
    trainer = Trainer(combined_model, (X_train_sup, y_train_sup), (X_val_sup, y_val_sup), config)

    # Run SSL pretraining.
    trainer.train_ssl_pretrainer(ssl_dataset)

    # Run supervised classifier training.
    trainer.train_classifier()

    # Example prediction with MC dropout inference.
    combined_model.eval()
    test_sample = X_val_sup.to(trainer.device)
    mean_pred, uncertainty = combined_model.predict(test_sample, mc_passes=config["training"]["mc_dropout_passes"])
    logger.info("MC Dropout Inference -- Mean Predictions: %s, Uncertainty: %s", mean_pred.detach().cpu().numpy(), uncertainty.detach().cpu().numpy())
