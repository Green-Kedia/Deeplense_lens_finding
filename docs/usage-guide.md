# DeepLense Usage Guide

This guide provides detailed instructions on using the DeepLense gravitational lens classification system, including training models, evaluating performance, and implementing ensemble methods.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Configuration](#configuration)
3. [Training Models](#training-models)
4. [Evaluation](#evaluation)
5. [Ensemble Methods](#ensemble-methods)
6. [Experimental Results](#experimental-results)
7. [Best Practices](#best-practices)

## Data Preparation

The dataset consists of astronomical images in three different filters for each object, with shape (3, 64, 64). The data should be organized in the following structure:

```
data/
├── train_lenses/       # Training lens images (.npy files)
├── train_nonlenses/    # Training non-lens images (.npy files)
├── test_lenses/        # Test lens images (.npy files)
└── test_nonlenses/     # Test non-lens images (.npy files)
```

No additional preprocessing is required as the data loading pipeline handles normalization and augmentation.

## Configuration

The system uses a configuration dictionary in `config.py` that controls all aspects of training and evaluation. Key parameters include:

```python
config = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'val_split': 0.2,
    'model_name': 'resnet50',  # Options: 'resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', etc.
    'loss_function': 'bce',    # Options: 'bce', 'focal', 'tversky', 'focaltversky', 'pin'
    'augmentation_level': 'medium',  # Options: 'none', 'light', 'medium', 'heavy'
}
```

You can modify these parameters directly in the config file or override them with command-line arguments.

## Training Models

### Basic Training

To train a model with default settings:

```bash
python main.py --mode train
```

### Customized Training

To customize training parameters:

```bash
python main.py --mode train --model efficientnet_b0 --batch_size 64 --epochs 100 --lr 0.0005 --augmentation heavy --loss focal
```

### Available Models

- `resnet18`, `resnet50`: ResNet architectures
- `efficientnet_b0`: EfficientNet B0
- `densenet121`: DenseNet-121
- `efficient_attention`: EfficientNet with physics-guided attention
- `efficient_fft_attention`: EfficientNet with FFT-based attention
- `fftaugmented`: Model with frequency domain feature augmentation
- `crop_view`: Multi-scale approach with center crop view
- `custom_cnn`: Custom CNN architecture

### Augmentation Levels

- `none`: No augmentation, only normalization
- `light`: Horizontal flips + normalization
- `medium`: Horizontal/vertical flips + rotation + normalization
- `heavy`: Flips + rotation + color jitter + normalization

### Loss Functions

- `bce`: Binary Cross Entropy loss (default)
- `focal`: Focal loss for class imbalance (alpha and gamma configurable)
- `tversky`: Tversky loss for better precision-recall balance
- `focaltversky`: Focal Tversky loss combining both approaches
- `pin`: Physics-informed loss with Einstein radius constraint

## Evaluation

### Evaluating a Saved Model

To evaluate a previously saved model:

```bash
python main.py --mode evaluate --model_path saved_models/resnet50_20250324-120000_model.pth --metadata_path saved_models/resnet50_20250324-120000_metadata.pth
```

The evaluation will output metrics including accuracy, precision, recall, F1 score, and ROC AUC, along with visualizations of the confusion matrix and ROC curve.

### Visualizing Data

To visualize samples from the dataset:

```bash
python main.py --mode visualize
```

## Ensemble Methods

Our best results were achieved using ensemble methods, combining predictions from multiple trained models.

### Creating an Ensemble

To evaluate using an ensemble of models:

```bash
python main.py --mode evaluate --ensemble \
  --model_paths saved_models/resnet50_model.pth saved_models/efficientnet_model.pth saved_models/efficientnet_modified_model.pth \
  --metadata_paths saved_models/resnet50_metadata.pth saved_models/efficientnet_metadata.pth saved_models/efficientnet_modified_metadata.pth \
  --ensemble_weights 1.0 1.0 1.0
```

The ensemble takes the weighted average of predictions from each model. If `--ensemble_weights` is not provided, equal weighting is used.

## Experimental Results

### Model Performance Comparison

| Model Configuration | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------| 
| Ensemble (ResNet50+EfficientNet+EfficientNet ModifiedReLU) | 0.9943 | 0.6694 | 0.8410 | 0.7455 | 0.9932 |
| EfficientNet B0 + Modified ReLU | 0.9944 | 0.6856 | 0.8051 | 0.7406 | 0.9877 |
| ResNet50 + Heavy Augmentation | 0.9937 | 0.6449 | 0.8103 | 0.7182 | 0.9914 |
| EfficientNet B0 + Modified ReLU (50 epochs) | 0.9929 | 0.5972 | 0.8667 | 0.7071 | 0.9909 |
| EfficientNet B0 + Heavy Augmentation | 0.9919 | 0.5612 | 0.8462 | 0.6748 | 0.9897 |
| EfficientNet B0 (50 epochs) | 0.9920 | 0.5640 | 0.8359 | 0.6736 | 0.9912 |
| ResNet50 (50 epochs) | 0.9916 | 0.5472 | 0.8615 | 0.6693 | 0.9910 |
| DenseNet121 | 0.9922 | 0.5768 | 0.7897 | 0.6667 | 0.9845 |
| EfficientNet B0 + FFT Attention | 0.9888 | 0.4659 | 0.8769 | 0.6085 | 0.9887 |
| EfficientNet B0 + Attention (PIN loss) | 0.9890 | 0.4683 | 0.7949 | 0.5894 | 0.9817 |
| EfficientNet B0 (Standard) | 0.9897 | 0.4886 | 0.8769 | 0.6275 | 0.9888 |
| ResNet18 + Medium Augmentation | 0.9900 | 0.4967 | 0.7641 | 0.6020 | 0.9803 |

### Ablation Study Results

| Configuration | Accuracy | Precision | Recall | F1 Score | ROC AUC | Change from Baseline |
|---------------|----------|-----------|--------|----------|---------|---------------------|
| ResNet18 (No Augmentation) | 0.9828 | 0.3386 | 0.7641 | 0.4693 | 0.9478 | Baseline |
| + Medium Augmentation | 0.9900 | 0.4967 | 0.7641 | 0.6020 | 0.9803 | +0.1327 F1, +0.0325 AUC |
| Switch to DenseNet121 | 0.9922 | 0.5768 | 0.7897 | 0.6667 | 0.9845 | +0.1974 F1, +0.0367 AUC |
| Switch to EfficientNet B0 | 0.9897 | 0.4886 | 0.8769 | 0.6275 | 0.9888 | +0.1582 F1, +0.0410 AUC |
| EfficientNet B0 + Heavy Aug | 0.9919 | 0.5612 | 0.8462 | 0.6748 | 0.9897 | +0.2055 F1, +0.0419 AUC |
| EfficientNet B0 + Modified ReLU | 0.9944 | 0.6856 | 0.8051 | 0.7406 | 0.9877 | +0.2713 F1, +0.0399 AUC |
| Switch to ResNet50 + Heavy Aug | 0.9937 | 0.6449 | 0.8103 | 0.7182 | 0.9914 | +0.2489 F1, +0.0436 AUC |
| Ensemble Approach | 0.9943 | 0.6694 | 0.8410 | 0.7455 | 0.9932 | +0.2762 F1, +0.0454 AUC |

### Architecture Comparison

| Architecture | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------|----------|-----------|--------|----------|---------| 
| ResNet18 | 0.9900 | 0.4967 | 0.7641 | 0.6020 | 0.9803 |
| Custom CNN | 0.9863 | 0.3972 | 0.7333 | 0.5153 | 0.9683 |
| DenseNet121 | 0.9922 | 0.5768 | 0.7897 | 0.6667 | 0.9845 |
| EfficientNet B0 | 0.9897 | 0.4886 | 0.8769 | 0.6275 | 0.9888 |
| ResNet50 | 0.9937 | 0.6449 | 0.8103 | 0.7182 | 0.9914 |

### Impact of Training Duration

| Model | Epochs | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|--------|----------|-----------|--------|----------|---------| 
| EfficientNet B0 | ~30 | 0.9897 | 0.4886 | 0.8769 | 0.6275 | 0.9888 |
| EfficientNet B0 | 50 | 0.9920 | 0.5640 | 0.8359 | 0.6736 | 0.9912 |
| ResNet50 | 50 | 0.9916 | 0.5472 | 0.8615 | 0.6693 | 0.9910 |
| EfficientNet B0 + Modified ReLU | 50 | 0.9929 | 0.5972 | 0.8667 | 0.7071 | 0.9909 |

### Special Case Analysis

| Experiment | Accuracy | Precision | Recall | F1 Score | ROC AUC | Notes |
|------------|----------|-----------|--------|----------|---------|-------|
| Weighted Sampling (EfficientNet) | 0.9486 | 0.1555 | 0.9436 | 0.2671 | 0.9872 | Extreme recall focus |
| Focal Loss | 0.9561 | 0.1740 | 0.9128 | 0.2923 | 0.9814 | Similar pattern to weighted sampling |
| FFT Augmented Model | 0.9890 | 0.4708 | 0.8667 | 0.6101 | 0.9866 | Frequency domain features |
| EfficientNet + FFT Attention | 0.9888 | 0.4659 | 0.8769 | 0.6085 | 0.9887 | Attention guided by FFT |
| Crop View | 0.9896 | 0.4843 | 0.7897 | 0.6004 | 0.9806 | Multi-scale approach |

## Best Practices

Based on our extensive experiments, we recommend the following best practices for gravitational lens classification:

1. **Model Selection**: 
   - ResNet50 and EfficientNet B0 provided the best balance of performance and efficiency
   - Consider the Modified ReLU activation for EfficientNet, which improved performance

2. **Data Augmentation**:
   - Heavy augmentation generally improves performance for all models
   - Include rotations since lensing artifacts can appear at any orientation

3. **Training Strategy**:
   - Early stopping with patience of 10 epochs works well
   - Aim for 50 epochs with cosine annealing learning rate schedule
   - Adam optimizer with initial learning rate of 0.001

4. **Handling Class Imbalance**:
   - Standard BCE loss with appropriate augmentation works well
   - If recall is critical, consider Focal Loss or weighted sampling, but expect precision to drop

5. **Ensemble Methods**:
   - Combine different architecture families (e.g., ResNet + EfficientNet)
   - Include models with different training strategies
   - Equal weighting works well in most cases

For production use, we recommend the ensemble approach (ResNet50 + EfficientNet B0 + EfficientNet B0 with Modified ReLU) as it achieved the best overall performance with F1 score of 0.7455 and ROC AUC of 0.9932.
