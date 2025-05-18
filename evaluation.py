"""evaluation.py

This module implements the Evaluation class for assessing the performance of the trained
CombinedModel on the test dataset. It computes evaluation metrics including F1-Score,
Matthews Correlation Coefficient (MCC), AUROC, AUPRC, Balanced Accuracy, and Accuracy.
It also implements an ablation study with input sparsity and an explainability method via
Integrated Gradients.

The Evaluation class adheres to the design:
  - __init__(self, model: CombinedModel, test_data: Any, config: dict)
  - evaluate_metrics(self) -> dict
  - run_ablation(self, drop_ratio: float = 0.5) -> dict
  - explain_sample(self, sample: torch.Tensor, baseline: torch.Tensor = None) -> torch.Tensor

Usage:
  Instantiate Evaluation with the trained CombinedModel, test dataset, and configuration.
  Then call the methods to compute metrics and integrated gradients explanations.
"""

import logging
import numpy as np
from typing import Any, Dict, Tuple, List

import torch
from torch import Tensor

# Import evaluation metric functions from scikit-learn
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluation:
    """
    Evaluation class for computing performance metrics, running ablation studies,
    and computing integrated gradients for explainability.
    """

    def __init__(self, model: Any, test_data: Any, config: dict) -> None:
        """
        Initialize the Evaluation class.

        Args:
            model (CombinedModel): The trained CombinedModel (with frozen SSLPretrainer and fine-tuned Classifier).
            test_data (Any): The test dataset. Can be a DataLoader or a tuple (X_test, y_test) where
                             X_test is a torch.Tensor of shape [N, channels, H, W] and
                             y_test is a torch.Tensor of shape [N].
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.test_data = test_data
        self.config = config

        # Determine the number of MC dropout passes from config; default to 30 if not set.
        self.mc_dropout_passes = (
            config.get("training", {}).get("mc_dropout_passes", 30)
            if config.get("training", {}).get("mc_dropout_passes", None) is not None
            else 30
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set evaluation mode by default (MC dropout will be enabled in predict)

        logger.info("Evaluation initialized with mc_dropout_passes=%d on device=%s", self.mc_dropout_passes, self.device)

    def evaluate_metrics(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set using MC dropout inference, and compute evaluation metrics.

        Returns:
            dict: Dictionary containing metrics:
                  - F1_score, Matthews_Correlation_Coefficient, AUROC, AUPRC,
                    Balanced_Accuracy, and Accuracy.
        """
        all_gt: List[int] = []
        all_pred_binary: List[int] = []
        all_pred_prob: List[float] = []

        # If test_data is a tuple (X_test, y_test) then unpack; otherwise assume DataLoader interface.
        if isinstance(self.test_data, tuple):
            X_test, y_test = self.test_data
            num_samples = X_test.size(0)
            # Process batch-wise (here, process whole tensor if fits in memory)
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            with torch.no_grad():
                mean_preds, _ = self.model.predict(X_test, mc_passes=self.mc_dropout_passes)
            # Convert predictions and ground truth to numpy arrays.
            pred_probs = mean_preds.squeeze(1).cpu().numpy()  # shape: [N]
            gt = y_test.cpu().numpy().astype(int)
            all_pred_prob.extend(pred_probs.tolist())
            # Threshold at 0.5 for binary decision.
            pred_binary = (pred_probs >= 0.5).astype(int)
            all_pred_binary.extend(pred_binary.tolist())
            all_gt.extend(gt.tolist())
        else:
            # Assume test_data is a DataLoader.
            for batch in self.test_data:
                # Expect batch as (x, y)
                x_batch, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                with torch.no_grad():
                    mean_preds, _ = self.model.predict(x_batch, mc_passes=self.mc_dropout_passes)
                # Convert predictions and labels.
                pred_probs = mean_preds.squeeze(1).cpu().numpy()
                gt = y_batch.cpu().numpy().astype(int)
                all_pred_prob.extend(pred_probs.tolist())
                pred_binary = (pred_probs >= 0.5).astype(int)
                all_pred_binary.extend(pred_binary.tolist())
                all_gt.extend(gt.tolist())

        # Compute metrics using scikit-learn functions, with safeguards.
        try:
            f1 = f1_score(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("F1 score computation error: %s", e)
            f1 = 0.0

        try:
            mcc = matthews_corrcoef(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("MCC computation error: %s", e)
            mcc = 0.0

        try:
            auroc = roc_auc_score(all_gt, all_pred_prob)
        except Exception as e:
            logger.warning("AUROC computation error: %s", e)
            auroc = 0.0

        try:
            auprc = average_precision_score(all_gt, all_pred_prob)
        except Exception as e:
            logger.warning("AUPRC computation error: %s", e)
            auprc = 0.0

        try:
            bal_acc = balanced_accuracy_score(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("Balanced Accuracy computation error: %s", e)
            bal_acc = 0.0

        try:
            acc = accuracy_score(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("Accuracy computation error: %s", e)
            acc = 0.0

        metrics = {
            "F1_score": f1,
            "Matthews_Correlation_Coefficient": mcc,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Balanced_Accuracy": bal_acc,
            "Accuracy": acc,
        }
        logger.info("Evaluation Metrics: %s", metrics)
        return metrics

    def run_ablation(self, drop_ratio: float = 0.5) -> Dict[str, float]:
        """
        Run an ablation study by applying input sparsity to the test samples.
        A proportion (drop_ratio) of the input features are set to zero to simulate missing data.
        The same metrics are computed on the sparsified test set.

        Args:
            drop_ratio (float): Fraction of input features to drop (set to zero). Default is 0.5 (50%).

        Returns:
            dict: Dictionary containing evaluation metrics on sparsified inputs.
        """
        all_gt: List[int] = []
        all_pred_binary: List[int] = []
        all_pred_prob: List[float] = []

        # Function to apply random dropout to a tensor
        def apply_feature_dropout(x: Tensor, ratio: float) -> Tensor:
            # Create a mask with ones and zeros; with probability (ratio) set to zero.
            dropout_mask = torch.bernoulli((1 - ratio) * torch.ones_like(x))
            # Multiply the input by the dropout mask.
            return x * dropout_mask

        if isinstance(self.test_data, tuple):
            X_test, y_test = self.test_data
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            # Create sparsified version of X_test.
            X_test_sparse = apply_feature_dropout(X_test, drop_ratio)
            with torch.no_grad():
                mean_preds, _ = self.model.predict(X_test_sparse, mc_passes=self.mc_dropout_passes)
            pred_probs = mean_preds.squeeze(1).cpu().numpy()
            gt = y_test.cpu().numpy().astype(int)
            all_pred_prob.extend(pred_probs.tolist())
            pred_binary = (pred_probs >= 0.5).astype(int)
            all_pred_binary.extend(pred_binary.tolist())
            all_gt.extend(gt.tolist())
        else:
            # Assume test_data is a DataLoader.
            for batch in self.test_data:
                x_batch, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Create sparsified input.
                x_batch_sparse = apply_feature_dropout(x_batch, drop_ratio)
                with torch.no_grad():
                    mean_preds, _ = self.model.predict(x_batch_sparse, mc_passes=self.mc_dropout_passes)
                pred_probs = mean_preds.squeeze(1).cpu().numpy()
                gt = y_batch.cpu().numpy().astype(int)
                all_pred_prob.extend(pred_probs.tolist())
                pred_binary = (pred_probs >= 0.5).astype(int)
                all_pred_binary.extend(pred_binary.tolist())
                all_gt.extend(gt.tolist())

        # Compute evaluation metrics on sparsified inputs.
        try:
            f1 = f1_score(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("F1 score computation error during ablation: %s", e)
            f1 = 0.0

        try:
            mcc = matthews_corrcoef(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("MCC computation error during ablation: %s", e)
            mcc = 0.0

        try:
            auroc = roc_auc_score(all_gt, all_pred_prob)
        except Exception as e:
            logger.warning("AUROC computation error during ablation: %s", e)
            auroc = 0.0

        try:
            auprc = average_precision_score(all_gt, all_pred_prob)
        except Exception as e:
            logger.warning("AUPRC computation error during ablation: %s", e)
            auprc = 0.0

        try:
            bal_acc = balanced_accuracy_score(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("Balanced Accuracy computation error during ablation: %s", e)
            bal_acc = 0.0

        try:
            acc = accuracy_score(all_gt, all_pred_binary)
        except Exception as e:
            logger.warning("Accuracy computation error during ablation: %s", e)
            acc = 0.0

        ablation_metrics = {
            "F1_score": f1,
            "Matthews_Correlation_Coefficient": mcc,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Balanced_Accuracy": bal_acc,
            "Accuracy": acc,
            "Drop_Ratio": drop_ratio
        }
        logger.info("Ablation Metrics (drop_ratio=%.2f): %s", drop_ratio, ablation_metrics)
        return ablation_metrics

    def explain_sample(self, sample: Tensor, baseline: Tensor = None, steps: int = 50) -> Tensor:
        """
        Compute Integrated Gradients (IG) for a given sample using the model's explain method.
        
        Args:
            sample (Tensor): Input sample tensor of shape [1, channels, height, width].
            baseline (Tensor, optional): Baseline tensor of the same shape as sample.
                                         If None, a zero-tensor is used.
            steps (int): Number of steps in the IG interpolation (default: 50).

        Returns:
            Tensor: Integrated gradients attribution tensor of shape [1, channels, height, width].
        """
        sample = sample.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(sample).to(self.device)
        else:
            baseline = baseline.to(self.device)
        self.model.eval()  # Ensure model is in eval mode for explanation.
        # Compute integrated gradients using the model's explain() method.
        with torch.no_grad():
            attributions = self.model.explain(sample, baseline, steps=steps)
        # Detach and move to CPU for further processing or visualization.
        attributions = attributions.detach().cpu()
        logger.info("Integrated Gradients calculated for the given sample.")
        return attributions


# Example usage (this block is for demonstration and testing purposes and
# should be removed or guarded by if __name__ == "__main__": in production):
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    # Dummy data and dummy model for demonstration.
    # In practice, the model would be an instance of CombinedModel already trained.
    dummy_X = torch.randn(20, 3, 64, 64)  # 20 test samples, 3 channels, 64x64 patches
    dummy_y = (torch.rand(20) > 0.5).long()
    test_dataset = TensorDataset(dummy_X, dummy_y)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Dummy model with a predict and explain method.
    class DummyCombinedModel(torch.nn.Module):
        def __init__(self):
            super(DummyCombinedModel, self).__init__()

        def predict(self, x: Tensor, mc_passes: int = 30) -> Tuple[Tensor, Tensor]:
            # Returns a constant prediction probability of 0.6 and uncertainty of 0.1
            batch_size = x.size(0)
            pred = 0.6 * torch.ones(batch_size, 1, device=x.device)
            uncert = 0.1 * torch.ones(batch_size, 1, device=x.device)
            return pred, uncert

        def explain(self, x: Tensor, baseline: Tensor, steps: int = 50) -> Tensor:
            # Returns a dummy integrated gradients attribution: difference between input and baseline.
            return x - baseline

        def to(self, device):
            return self

        def eval(self):
            return self

    dummy_model = DummyCombinedModel()

    # Dummy configuration dictionary (simulate loaded config.yaml)
    dummy_config = {
        "training": {
            "mc_dropout_passes": 30
        },
        "data": {
            "num_channels": 3,
            "patch_window_size": 64,
            "mask_ratio": 0.75,
            "undersample_filter_ratio": 0.05
        }
    }

    evaluator = Evaluation(model=dummy_model, test_data=test_loader, config=dummy_config)
    metrics = evaluator.evaluate_metrics()
    print("Evaluation Metrics:", metrics)
    ablation_metrics = evaluator.run_ablation(drop_ratio=0.5)
    print("Ablation Metrics:", ablation_metrics)
    # Run explanation on one sample.
    sample = dummy_X[0].unsqueeze(0)  # single sample with shape [1, 3, 64, 64]
    attribution = evaluator.explain_sample(sample)
    print("Integrated Gradients Attribution shape:", attribution.shape)
"""