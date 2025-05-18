"""dataset_loader.py

This module implements the DatasetLoader class for loading and preprocessing
geospatial raster data (explanatory features and label rasters), slicing the rasters
into patches, and splitting the resulting dataset into training, validation, and test sets.
"""

import os
import logging
import numpy as np
import rasterio
import geopandas as gpd
from typing import Tuple, List, Any, Union
from scipy.ndimage import gaussian_filter
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """DatasetLoader class for handling geospatial data loading and preprocessing.

    Methods:
        load_geospatial_data() -> Tuple[np.ndarray, np.ndarray]:
            Loads and returns the raw explanatory raster X_raw and label raster Y_raw.
        preprocess_data(X_raw: np.ndarray) -> np.ndarray:
            Performs outlier removal, imputation, and normalization on X_raw.
        create_patches(X: np.ndarray, w: int) -> List[np.ndarray]:
            Slices the processed raster X into a list of patches of size [m, w, w].
        split_data(samples: np.ndarray, labels: np.ndarray, seed: int = 42) -> Tuple[Any, Any, Any]:
            Splits the samples and labels into training, validation, and test sets.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize DatasetLoader with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing file paths and data parameters.
        """
        self.config = config
        self.data_config = config.get("data", {})
        # Get file paths for explanatory raster and label rasters
        self.explanatory_raster_path: str = self.data_config.get("explanatory_raster_path", "data/explanatory.tif")
        # Label raster paths can be a list of file paths; if not provided, default to a single file.
        self.label_raster_paths: Union[List[str], str] = self.data_config.get("label_raster_paths", "data/label.tif")

        # Get numerical hyperparameters with default values if not specified.
        self.num_channels: int = self.data_config.get("num_channels", 3)  # default 3 channels
        self.patch_window_size: int = self.data_config.get("patch_window_size", 64)  # default 64x64 patch
        self.mask_ratio: float = self.data_config.get("mask_ratio", 0.75)
        self.undersample_filter_ratio: float = self.data_config.get("undersample_filter_ratio", 0.05)

        # Set a default random seed for reproducibility
        self.random_seed: int = self.config.get("random_seed", 42)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Placeholders for the loaded raw data.
        self.X_raw: np.ndarray = None  # Explanatory raster, shape: (m, r, c)
        self.Y_raw: np.ndarray = None  # Label raster, shape: (1, r, c) or (r, c)

        logger.info("DatasetLoader initialized with config: %s", self.config)

    def load_geospatial_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the raw geospatial raster data for explanatory features (X_raw) and labels (Y_raw).

        Returns:
            Tuple containing:
                - X_raw (np.ndarray): Multi-band explanatory raster of shape (m, rows, cols).
                - Y_raw (np.ndarray): Label raster of shape (rows, cols) or (1, rows, cols).
        """
        # Check if explanatory raster file exists
        if not os.path.exists(self.explanatory_raster_path):
            logger.error("Explanatory raster file does not exist: %s", self.explanatory_raster_path)
            raise FileNotFoundError(f"Explanatory raster file not found: {self.explanatory_raster_path}")

        # Read explanatory raster with rasterio
        with rasterio.open(self.explanatory_raster_path) as src:
            X_raw = src.read()  # shape: (bands, rows, cols)
            src_crs = src.crs  # Coordinate Reference System
            logger.info("Loaded explanatory raster with shape %s and CRS %s", X_raw.shape, src_crs)

        # Check if label raster paths is a list or a single path.
        if isinstance(self.label_raster_paths, list):
            # If multiple label raster files are provided, load them and stack along a new axis.
            Y_list = []
            for label_path in self.label_raster_paths:
                if not os.path.exists(label_path):
                    logger.error("Label raster file does not exist: %s", label_path)
                    raise FileNotFoundError(f"Label raster file not found: {label_path}")
                with rasterio.open(label_path) as src_label:
                    y = src_label.read(1)  # read the first band, shape: (rows, cols)
                    Y_list.append(y)
            # Stack label rasters along the channel axis
            Y_raw = np.stack(Y_list, axis=0)
            logger.info("Loaded %d label rasters with shape %s", len(Y_list), Y_raw.shape)
        else:
            # Single label raster file provided as string.
            if not os.path.exists(self.label_raster_paths):
                logger.error("Label raster file does not exist: %s", self.label_raster_paths)
                raise FileNotFoundError(f"Label raster file not found: {self.label_raster_paths}")
            with rasterio.open(self.label_raster_paths) as src_label:
                Y_raw = src_label.read(1)  # shape: (rows, cols)
                logger.info("Loaded label raster with shape %s", Y_raw.shape)

        # Here we assume that CRS for explanatory and label rasters are the same.
        # If not, they must be reprojected. That functionality is not implemented here.

        self.X_raw = X_raw
        self.Y_raw = Y_raw
        return X_raw, Y_raw

    def preprocess_data(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Preprocess the explanatory raster data:
          - Outlier removal using Tukey fences.
          - Imputation of missing values using inverse distance weighting (IDW) approximation.
          - Smoothing with a Gaussian filter.
          - Normalization with z-score.

        Args:
            X_raw (np.ndarray): Raw multi-band raster data of shape (m, rows, cols).

        Returns:
            np.ndarray: Processed raster data with the same shape as X_raw.
        """
        X_processed = X_raw.astype(np.float32)
        m, rows, cols = X_processed.shape
        logger.info("Starting preprocessing on data with shape: %s", X_processed.shape)

        # For each channel, remove outliers and impute missing values
        for ch in range(m):
            channel = X_processed[ch]
            # Compute quartiles and interquartile range (IQR)
            q1 = np.percentile(channel, 25)
            q3 = np.percentile(channel, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Mark outliers as NaN
            outliers = (channel < lower_bound) | (channel > upper_bound)
            num_outliers = np.sum(outliers)
            if num_outliers > 0:
                logger.info("Channel %d: Found %d outliers", ch, num_outliers)
            channel[outliers] = np.nan

            # Impute missing values (NaN) with the median of non-NaN values
            median_val = np.nanmedian(channel)
            inds_nan = np.isnan(channel)
            channel[inds_nan] = median_val

            # Apply Gaussian smoothing filter
            channel_smoothed = gaussian_filter(channel, sigma=1)
            X_processed[ch] = channel_smoothed

        # Normalize each channel using z-score normalization
        for ch in range(m):
            channel = X_processed[ch]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            if std_val == 0:
                logger.warning("Channel %d has zero standard deviation during normalization.", ch)
                std_val = 1.0
            X_processed[ch] = (channel - mean_val) / std_val
            logger.info("Channel %d normalized: mean=%.3f, std=%.3f", ch, mean_val, std_val)

        logger.info("Preprocessing complete. Processed data shape: %s", X_processed.shape)
        return X_processed

    def create_patches(self, X: np.ndarray, w: int) -> List[np.ndarray]:
        """
        Slice the multi-band raster X into non-overlapping square patches of size (m, w, w).

        Args:
            X (np.ndarray): Processed raster data of shape (m, rows, cols).
            w (int): Patch/window size.

        Returns:
            List[np.ndarray]: List of patches, each with shape (m, w, w).
        """
        m, rows, cols = X.shape
        patches = []
        # Determine number of patches along rows and columns.
        num_patches_row = rows // w
        num_patches_col = cols // w

        logger.info("Creating patches with window size %d, total patches: %d x %d", w, num_patches_row, num_patches_col)

        for i in range(num_patches_row):
            for j in range(num_patches_col):
                patch = X[:, i * w:(i + 1) * w, j * w:(j + 1) * w]
                # Optionally, store patch location information if needed
                patches.append(patch)
        logger.info("Total patches created: %d", len(patches))
        return patches

    def extract_labels(self, Y: Union[np.ndarray, List[np.ndarray]], w: int) -> List[int]:
        """
        Extract labels for each patch by taking the center pixel label.
        Supports both single-label raster (2D array) and multi-label (3D array) formats.
        For multi-label, the label will be chosen from the first layer.

        Args:
            Y (np.ndarray or List[np.ndarray]): Label raster(s). If 2D, shape is (rows, cols).
                If 3D, shape is (channels, rows, cols).
            w (int): Patch/window size.

        Returns:
            List[int]: List of labels for each patch (extracted from the center pixel).
        """
        if isinstance(Y, list):
            # If Y is provided as a list of arrays, stack them into a single array along the first axis.
            Y = np.stack(Y, axis=0)

        if Y.ndim == 3:
            # Use the first channel for label extraction
            label_array = Y[0]
        elif Y.ndim == 2:
            label_array = Y
        else:
            logger.error("Label raster Y has unsupported dimensions: %s", Y.shape)
            raise ValueError("Unsupported label raster format.")

        rows, cols = label_array.shape
        labels = []
        num_patches_row = rows // w
        num_patches_col = cols // w

        logger.info("Extracting labels from label raster with shape %s", label_array.shape)
        for i in range(num_patches_row):
            for j in range(num_patches_col):
                # Get center pixel index of the patch
                center_row = i * w + w // 2
                center_col = j * w + w // 2
                label = label_array[center_row, center_col]
                labels.append(int(label))
        logger.info("Extracted %d labels from patches.", len(labels))
        return labels

    def split_data(
        self, samples: List[np.ndarray], labels: List[int], seed: int = 42
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Splits the dataset (samples and corresponding labels) into training, validation,
        and test sets with an 80/10/10 split.

        Args:
            samples (List[np.ndarray]): List of patch samples.
            labels (List[int]): List of labels corresponding to each patch.
            seed (int): Random seed for reproducibility (default: 42).

        Returns:
            Tuple containing:
              - (X_train, y_train): Training set
              - (X_val, y_val): Validation set
              - (X_test, y_test): Test set
        """
        np.random.seed(seed)
        total_samples = len(samples)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        logger.info("Shuffled data indices with seed %d", seed)

        # Determine split sizes
        train_end = int(0.8 * total_samples)
        val_end = int(0.9 * total_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        X_train = np.array([samples[i] for i in train_indices])
        y_train = np.array([labels[i] for i in train_indices])
        X_val = np.array([samples[i] for i in val_indices])
        y_val = np.array([labels[i] for i in val_indices])
        X_test = np.array([samples[i] for i in test_indices])
        y_test = np.array([labels[i] for i in test_indices])

        logger.info("Data split into train (%d), val (%d), and test (%d) samples.",
                    len(train_indices), len(val_indices), len(test_indices))
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Example usage (would be removed or placed under a __main__ guard in actual project)
if __name__ == "__main__":
    # Example config dictionary that would be loaded from config.yaml
    example_config = {
        "data": {
            "explanatory_raster_path": "data/explanatory.tif",
            "label_raster_paths": "data/label.tif",
            "num_channels": 3,
            "patch_window_size": 64,
            "mask_ratio": 0.75,
            "undersample_filter_ratio": 0.05
        },
        "random_seed": 42
    }

    loader = DatasetLoader(example_config)
    try:
        X_raw, Y_raw = loader.load_geospatial_data()
        X_processed = loader.preprocess_data(X_raw)
        patches = loader.create_patches(X_processed, loader.patch_window_size)
        labels = loader.extract_labels(Y_raw, loader.patch_window_size)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data(patches, labels, seed=42)
        logger.info("Data pipeline complete. Training samples: %d, Validation samples: %d, Test samples: %d",
                    X_train.shape[0], X_val.shape[0], X_test.shape[0])
    except Exception as e:
        logger.error("Error during data loading: %s", e)
