# Enabling Scalable Mineral Exploration: Self-Supervision and Explainability

This repository contains an implementation of a self-supervised learning approach for mineral prospectivity mapping using Vision Transformers and explainable AI techniques.

**Note**: This is an automated Paper2Code implementation. The original paper can be found at: https://link.springer.com/article/10.1007/s12145-025-01718-y

## Overview

The pipeline implements:
- **Self-Supervised Pretraining**: Vision Transformer (ViT) based encoder-decoder with masked image modeling
- **Supervised Fine-tuning**: Multi-layer perceptron classifier with undersampling strategy
- **Uncertainty Estimation**: Monte Carlo dropout for epistemic uncertainty quantification
- **Explainability**: Integrated Gradients for model interpretation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RichardScottOZ/Enabling_Scalable_Mineral_Exploration_Self-Supervision_and_Explainability.git
cd Enabling_Scalable_Mineral_Exploration_Self-Supervision_and_Explainability
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support with CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data Preparation

The pipeline expects geospatial raster data in GeoTIFF format:

1. **Explanatory Raster**: Multi-band raster containing geological/geophysical features
   - Format: `.tif` or `.tiff`
   - Shape: (num_channels, height, width)
   - Example channels: magnetic anomaly, gravity, radiometric data, etc.

2. **Label Raster**: Binary labels indicating mineral presence/absence
   - Format: `.tif` or `.tiff`
   - Shape: (height, width) or (1, height, width)
   - Values: 0 (unknown/absent) or 1 (present)

3. Place your data files in a `data/` directory (or update paths in `config.yaml`):
```
data/
├── explanatory.tif  # Your multi-band feature raster
└── label.tif        # Your binary label raster
```

### Data Requirements
- Rasters must have the same coordinate reference system (CRS)
- Rasters must have the same spatial extent and resolution
- Raster dimensions should be divisible by the patch window size (default: 64)

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
training:
  learning_rate: 0.001        # Learning rate for optimization
  batch_size: 32              # Training batch size
  pretraining_epochs: 30      # SSL pretraining epochs
  supervised_epochs: 10       # Supervised fine-tuning epochs
  mc_dropout_passes: 30       # Monte Carlo dropout passes

model:
  encoder:
    patch_size: 16            # Patch size for Vision Transformer
    num_layers: 6             # Number of transformer layers
    hidden_dim: 128           # Hidden dimension
  classifier:
    dropout: 0.1              # Dropout rate

data:
  num_channels: 3             # Number of input channels
  patch_window_size: 64       # Window size for patches
  mask_ratio: 0.75            # Masking ratio for SSL (75%)
  explanatory_raster_path: "data/explanatory.tif"
  label_raster_paths: "data/label.tif"
```

## Usage

### Basic Usage

Run the complete pipeline with default settings:
```bash
python main.py
```

### Advanced Usage

Specify a custom configuration file:
```bash
python main.py --config my_config.yaml
```

Skip SSL pretraining if encoder weights already exist:
```bash
python main.py --skip-ssl-training --ssl-weights ssl_pretrained_encoder.pth
```

### Command Line Arguments

- `--config PATH`: Path to configuration YAML file (default: `config.yaml`)
- `--skip-ssl-training`: Skip SSL pretraining phase
- `--ssl-weights PATH`: Path to load/save SSL encoder weights (default: `ssl_pretrained_encoder.pth`)
- `--classifier-weights PATH`: Path to load/save classifier weights (default: `supervised_classifier.pth`)

## Pipeline Stages

1. **Data Loading**: Loads and validates geospatial rasters
2. **Preprocessing**: Outlier removal, imputation, smoothing, normalization
3. **Patch Creation**: Slices rasters into fixed-size patches
4. **Data Splitting**: 80% train, 10% validation, 10% test
5. **SSL Pretraining**: Trains encoder-decoder with masked image modeling
6. **Supervised Training**: Trains classifier with undersampling strategy
7. **Evaluation**: Computes metrics (F1, MCC, AUROC, AUPRC, Balanced Accuracy, Accuracy)
8. **Ablation Study**: Tests robustness with feature dropout
9. **Explainability**: Computes Integrated Gradients attributions

## Output

The pipeline generates:
- `ssl_pretrained_encoder.pth`: Pretrained encoder weights
- `supervised_classifier.pth`: Trained classifier weights
- Console logs with training progress and evaluation metrics
- Evaluation metrics printed at completion

### Example Output
```
==================================================
EVALUATION RESULTS
==================================================
F1_score................................ 0.8234
Matthews_Correlation_Coefficient........ 0.7891
AUROC................................... 0.9123
AUPRC................................... 0.8756
Balanced_Accuracy....................... 0.8567
Accuracy................................ 0.8901
==================================================
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'rasterio'**
- Solution: Install dependencies with `pip install -r requirements.txt`

**FileNotFoundError: data/explanatory.tif not found**
- Solution: Place your data files in the `data/` directory or update paths in `config.yaml`

**CUDA out of memory**
- Solution: Reduce `batch_size` in `config.yaml` (try 16 or 8)

**RuntimeError: Sizes of tensors must match**
- Solution: Ensure raster dimensions are divisible by `patch_window_size`

**Very low metric scores**
- Check that your label raster has both 0 and 1 values
- Verify that rasters are properly aligned
- Try adjusting `undersample_filter_ratio` in config

## Project Structure

```
.
├── main.py              # Main entry point
├── config.yaml          # Configuration file
├── dataset_loader.py    # Data loading and preprocessing
├── model.py             # Model architectures (SSLPretrainer, Classifier)
├── trainer.py           # Training logic
├── evaluation.py        # Evaluation and explainability
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Citation

If you use this code, please cite the original paper:

```
[Paper citation to be added - link: https://link.springer.com/article/10.1007/s12145-025-01718-y]
```

## Contributing

This is an automated Paper2Code implementation. Contributions to improve usability and robustness are welcome!

## License

Please refer to the original paper for licensing information.

## Acknowledgments

- Original implementation generated by Paper2Code: https://github.com/going-doer/Paper2Code
- This version has been enhanced for usability and robustness

