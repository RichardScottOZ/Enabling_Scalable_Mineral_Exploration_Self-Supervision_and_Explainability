"""model.py

This module implements PyTorch model classes for the self-supervised pretraining
and supervised fine-tuning phases for mineral prospectivity mapping.
It includes:
  - SSLPretrainer: Encoder-decoder architecture using a Vision Transformer encoder
    and a lightweight Transformer decoder, with masking support.
  - Classifier: A simple MLP with Parametric ReLU, BatchNorm, and Dropout for processing
    latent features extracted by the frozen encoder.
  - CombinedModel: Integration of the frozen SSLPretrainer encoder and the Classifier.
    It provides methods for forward prediction, Monte-Carlo dropout inference, and
    Integrated Gradients explainability.
    
Configuration values are expected to come from config.yaml.
Default values are provided when configuration parameters are missing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List

# ---------------------------
# Helper functions and defaults

def default_value(value, default):
    return value if value is not None else default

# ---------------------------
# SSLPretrainer Class

class SSLPretrainer(nn.Module):
    """
    SSLPretrainer implements a masked image modeling pretraining architecture.
    It uses a Vision Transformer (ViT) as the encoder and a lightweight transformer
    decoder for reconstructing masked patches.
    """
    def __init__(self, params: dict) -> None:
        """
        Initialize the SSLPretrainer.
        
        Args:
            params (dict): Dictionary configuration for model parameters.
                Expected keys:
                  - "encoder": dict, containing keys "architecture", "patch_size",
                               "num_layers", "hidden_dim".
                  - "decoder": dict, containing keys "architecture", "num_layers", "hidden_dim".
                  - "data": dict, containing "num_channels".
        """
        super(SSLPretrainer, self).__init__()
        
        # Extract encoder configuration from params
        encoder_config = params.get("encoder", {})
        self.architecture = default_value(encoder_config.get("architecture"), "Vision Transformer")
        self.patch_size = default_value(encoder_config.get("patch_size"), 16)
        self.enc_num_layers = default_value(encoder_config.get("num_layers"), 6)
        self.hidden_dim = default_value(encoder_config.get("hidden_dim"), 128)
        
        # Extract decoder configuration from params
        decoder_config = params.get("decoder", {})
        self.dec_num_layers = default_value(decoder_config.get("num_layers"), 2)
        self.dec_hidden_dim = default_value(decoder_config.get("hidden_dim"), self.hidden_dim)
        
        # Data configuration
        data_config = params.get("data", {})
        self.num_channels = default_value(data_config.get("num_channels"), 3)
        
        # For patch embedding, we use a linear projection.
        # The patch dimension is: channels * (patch_size * patch_size)
        self.patch_dim = self.num_channels * (self.patch_size ** 2)
        
        # A linear layer to embed flattened patches into hidden_dim tokens.
        self.patch_embed = nn.Linear(self.patch_dim, self.hidden_dim)
        
        # Calculate number of patches will be dynamic depending on input image dimensions.
        # Positional embeddings will be added later in forward once sequence length is known.
        self.pos_embed = None  # This will be initialized in forward if not already done.
        
        # Vision Transformer encoder: using standard TransformerEncoderLayer for simplicity.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=4, dim_feedforward=self.hidden_dim * 4, dropout=0.1, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.enc_num_layers)
        
        # Lightweight Transformer decoder: using nn.TransformerEncoder with small number of layers.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=4, dim_feedforward=self.dec_hidden_dim * 4, dropout=0.1, activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=self.dec_num_layers)
        
        # Reconstruction head: projects decoder token to reconstruction vector for the patch.
        self.recon_head = nn.Linear(self.hidden_dim, self.patch_dim)
        
        # Learnable mask token for masked patches (shape: [1, hidden_dim])
        self.mask_token = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.mask_token)
        
        # Initialize weights for patch_embed
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def mask_input(self, x: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor, int, int]:
        """
        Divide input images into non-overlapping patches, and randomly mask a proportion of patches.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
            mask_ratio (float): Ratio of patches to mask (e.g., 0.75).
        
        Returns:
            visible_tokens (Tensor): Tensor of shape [batch, num_visible, hidden_dim] for visible patches.
            mask (Tensor): Binary mask tensor of shape [batch, total_patches], where 0 indicates visible and 1 indicates masked.
            grid_h (int): Number of patches along height.
            grid_w (int): Number of patches along width.
        """
        batch_size, channels, img_h, img_w = x.size()
        device = x.device

        # Calculate grid size (number of patches)
        grid_h = img_h // self.patch_size
        grid_w = img_w // self.patch_size
        total_patches = grid_h * grid_w
        
        # Use nn.Unfold to extract patches: output shape [batch, patch_dim, num_patches]
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold(x)  # shape: [batch, patch_dim, total_patches]
        patches = patches.transpose(1, 2)  # shape: [batch, total_patches, patch_dim]
        
        # Embed patches: [batch, total_patches, hidden_dim]
        patch_tokens = self.patch_embed(patches)
        
        # Create a mask for patches: for each sample, randomly choose visible indices.
        visible_tokens_list = []
        mask_tensor = torch.zeros((batch_size, total_patches), dtype=torch.long, device=device)
        for b in range(batch_size):
            indices = torch.randperm(total_patches, device=device)
            num_visible = int(math.floor(total_patches * (1 - mask_ratio)))
            visible_idx = indices[:num_visible]
            mask = torch.ones(total_patches, device=device)
            mask[visible_idx] = 0  # 0 means visible, 1 means masked
            mask_tensor[b] = mask.long()
            visible_tokens_list.append(patch_tokens[b, visible_idx, :])
        # To simplify, pad visible tokens back to full sequence length later in forward.
        # Instead, we return the full patch_tokens and mask indicating which ones are visible.
        return patch_tokens, mask_tensor, grid_h, grid_w

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for self-supervised pretraining.
        It uses mask_input to select visible tokens, passes them through the encoder,
        then reconstructs the full patch sequence using the decoder, and finally reconstructs the image.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
        
        Returns:
            reconstructed_x (Tensor): Reconstructed image tensor of shape [batch, channels, height, width].
        """
        batch_size, channels, img_h, img_w = x.size()
        device = x.device

        # Extract patch tokens and mask
        patch_tokens, mask, grid_h, grid_w = self.mask_input(x, mask_ratio=default_value(self.training and self.patch_embed.training, 0.75))
        # Note: Passing mask_ratio from config; here we use self.training as indicator (but always use config value)
        # Instead, using the configured mask ratio from data config is more proper:
        mask_ratio = 0.75  # default, can be adjusted via config
        # Overwrite mask_input call with correct mask_ratio parameter from config if available.
        patch_tokens, mask, grid_h, grid_w = self.mask_input(x, mask_ratio=mask_ratio)
        total_patches = patch_tokens.size(1)  # total number of patches
        
        # For each batch, reconstruct the full token sequence using encoder output for visible tokens and mask token for masked.
        # First, create a placeholder for tokens: shape [batch, total_patches, hidden_dim]
        tokens_full = torch.zeros_like(patch_tokens, device=device)
        # We process each sample separately since visible tokens are at variable positions.
        for b in range(batch_size):
            visible_idx = (mask[b] == 0).nonzero(as_tuple=False).squeeze(1)
            # Get visible tokens for sample b
            visible_tokens = patch_tokens[b:b+1, visible_idx, :]  # shape: [1, num_visible, hidden_dim]
            # Pass visible tokens through encoder.
            # Note: Transformer expects sequence of shape [S, batch, d_model], so we transpose.
            visible_tokens_enc = self.encoder(visible_tokens.transpose(0, 1))  # shape: [num_visible, 1, hidden_dim]
            visible_tokens_enc = visible_tokens_enc.transpose(0, 1).squeeze(0)  # shape: [num_visible, hidden_dim]
            # Initialize full token sequence with mask tokens.
            tokens_sample = self.mask_token.repeat(total_patches, 1)  # shape: [total_patches, hidden_dim]
            # Replace positions of visible patches with encoder outputs.
            tokens_sample[visible_idx] = visible_tokens_enc
            tokens_full[b] = tokens_sample

        # Initialize positional embeddings if not done already.
        if self.pos_embed is None or self.pos_embed.size(1) != total_patches:
            self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, self.hidden_dim, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Add positional embeddings
        tokens_full = tokens_full + self.pos_embed

        # Pass full token sequence through decoder
        # Transformer expects [S, batch, d_model]
        tokens_dec = self.decoder(tokens_full.transpose(0, 1))  # shape: [total_patches, batch, hidden_dim]
        tokens_dec = tokens_dec.transpose(0, 1)  # shape: [batch, total_patches, hidden_dim]
        
        # Reconstruct each patch via reconstruction head
        patch_recon = self.recon_head(tokens_dec)  # shape: [batch, total_patches, patch_dim]
        
        # Reshape patches back to spatial grid and then fold them back to image.
        # First, reshape to (batch, total_patches, num_channels, patch_size, patch_size)
        patch_recon = patch_recon.view(batch_size, total_patches, self.num_channels, self.patch_size, self.patch_size)
        # Rearrange patches into image: For that, we can reshape and then permute.
        patch_recon = patch_recon.view(batch_size, grid_h, grid_w, self.num_channels, self.patch_size, self.patch_size)
        # Permute to (batch, num_channels, grid_h, patch_size, grid_w, patch_size)
        patch_recon = patch_recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Merge grid and patch dimensions to get (batch, num_channels, img_h, img_w)
        reconstructed_x = patch_recon.view(batch_size, self.num_channels, grid_h * self.patch_size, grid_w * self.patch_size)
        
        return reconstructed_x

    def compute_loss(self, reconstructed: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Mean Squared Error (MSE) reconstruction loss between the reconstructed image
        and the target image.
        
        Args:
            reconstructed (Tensor): Reconstructed image [batch, channels, height, width].
            target (Tensor): Original input image [batch, channels, height, width].
        
        Returns:
            Tensor: MSE loss.
        """
        loss = F.mse_loss(reconstructed, target)
        return loss

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract latent features from the input x using the full patch embedding and encoder (with all patches visible).
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
        
        Returns:
            Tensor: Latent features of shape [batch, hidden_dim] (averaged across patches).
        """
        batch_size, channels, img_h, img_w = x.size()
        device = x.device
        
        # Extract patches using Unfold
        unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold(x)  # shape: [batch, patch_dim, total_patches]
        patches = patches.transpose(1, 2)  # shape: [batch, total_patches, patch_dim]
        
        # Embed patches
        patch_tokens = self.patch_embed(patches)  # shape: [batch, total_patches, hidden_dim]
        
        # For feature extraction, use all patches (i.e., no masking)
        # Add positional embeddings; initialize if necessary.
        total_patches = patch_tokens.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != total_patches:
            self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, self.hidden_dim, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        tokens = patch_tokens + self.pos_embed

        # Pass through encoder; Transformer expects [S, batch, d_model]
        tokens_enc = self.encoder(tokens.transpose(0, 1))
        tokens_enc = tokens_enc.transpose(0, 1)  # shape: [batch, total_patches, hidden_dim]
        
        # Aggregate token features, e.g., by taking the mean over patches.
        features = tokens_enc.mean(dim=1)  # shape: [batch, hidden_dim]
        return features

# ---------------------------
# Classifier Class

class Classifier(nn.Module):
    """
    Classifier implements a simple Multi-Layer Perceptron (MLP) which takes the one-dimensional 
    latent features from the SSL encoder and outputs a scalar prediction estimating the mineral 
    presence likelihood.
    """
    def __init__(self, params: dict) -> None:
        """
        Initialize the Classifier.
        
        Args:
            params (dict): Dictionary configuration for the classifier parameters.
                Expected keys:
                  - "input_dim": Dimension of the input features (should match SSLPretrainer hidden_dim).
                  - "hidden_dims": List of hidden layer dimensions (optional).
                  - "dropout": Dropout probability.
                  - "activation": Activation to use (default: Parametric ReLU).
        """
        super(Classifier, self).__init__()
        # Read input dimension from params; default to 128 if not provided.
        self.input_dim = default_value(params.get("input_dim"), 128)
        hidden_dims = default_value(params.get("hidden_dims"), [self.input_dim // 2])
        self.dropout_rate = default_value(params.get("dropout"), 0.1)
        self.activation_name = default_value(params.get("activation"), "Parametric ReLU")
        
        layers = []
        in_dim = self.input_dim
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            if self.activation_name.lower() in ["prelu", "parametric relu"]:
                layers.append(nn.PReLU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h_dim
        
        # Final output layer: outputs a single scalar.
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights for linear layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, features: Tensor) -> Tensor:
        """
        Forward pass for the Classifier.
        
        Args:
            features (Tensor): Latent features of shape [batch, input_dim].
        
        Returns:
            Tensor: Predictions of shape [batch, 1] representing likelihood.
        """
        out = self.mlp(features)
        # Optionally, apply a sigmoid so that output is in [0,1]
        out = torch.sigmoid(out)
        return out

# ---------------------------
# CombinedModel Class

class CombinedModel(nn.Module):
    """
    CombinedModel integrates a frozen SSLPretrainer encoder with a Classifier.
    It provides methods for feature extraction, prediction with Monte-Carlo dropout inference,
    and integrated gradients explainability.
    """
    def __init__(self, encoder: SSLPretrainer, classifier: Classifier) -> None:
        """
        Initialize the CombinedModel.
        
        Args:
            encoder (SSLPretrainer): A pretrained SSLPretrainer model.
            classifier (Classifier): A classifier model.
        """
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract features from input x using the frozen encoder.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
            
        Returns:
            Tensor: Latent features of shape [batch, hidden_dim].
        """
        features = self.encoder.extract_features(x)
        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the combined model: extract features and pass through classifier.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
            
        Returns:
            Tensor: Predictions of shape [batch, 1].
        """
        features = self.extract_features(x)
        prediction = self.classifier(features)
        return prediction

    def predict(self, x: Tensor, mc_passes: int = 30) -> Tuple[Tensor, Tensor]:
        """
        Perform Monte-Carlo dropout inference to compute mean prediction and
        epistemic uncertainty.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
            mc_passes (int): Number of stochastic forward passes.
            
        Returns:
            Tuple[Tensor, Tensor]: Mean predictions and standard deviation (uncertainty)
                with shapes ([batch, 1], [batch, 1]).
        """
        self.classifier.train()  # Ensure dropout is active in classifier
        preds = []
        for _ in range(mc_passes):
            pred = self.forward(x)
            preds.append(pred)
        preds_stack = torch.stack(preds, dim=0)  # shape: [mc_passes, batch, 1]
        mean_pred = preds_stack.mean(dim=0)
        std_pred = preds_stack.std(dim=0)
        return mean_pred, std_pred

    def explain(self, x: Tensor, baseline: Tensor, steps: int = 50) -> Tensor:
        """
        Compute Integrated Gradients (IG) for the prediction with respect to the input x.
        The IG method interpolates between a baseline and the input and accumulates the gradients.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width].
            baseline (Tensor): Baseline tensor of the same shape as x.
            steps (int): Number of steps in the interpolation.
            
        Returns:
            Tensor: Integrated gradients attribution tensor of shape [batch, channels, height, width].
        """
        # Ensure baseline has same shape as input
        assert baseline.shape == x.shape, "Baseline and input must have the same shape."
        
        # Scale inputs and accumulate gradients
        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(0, steps + 1)]
        grads = []
        
        # Set up the model in evaluation mode but enable gradient computation for x.
        self.eval()
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.requires_grad_(True)
            # Forward pass: get prediction output, assume scalar per example.
            output = self.forward(scaled_input)
            # Sum outputs to get a scalar for backward.
            total_output = output.sum()
            self.zero_grad()
            total_output.backward(retain_graph=True)
            # Get gradients with respect to input.
            grad = scaled_input.grad.clone()
            grads.append(grad)
        
        # Approximate the integral using the trapezoidal rule.
        grads = torch.stack(grads, dim=0)  # shape: [steps+1, batch, channels, height, width]
        avg_grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = avg_grads.mean(dim=0)  # shape: [batch, channels, height, width]
        
        integrated_grads = (x - baseline) * avg_grads  # element-wise product
        return integrated_grads
