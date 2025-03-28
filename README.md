# DeepLense: Gravitational Lens Classification

This repository contains code for identifying gravitational lensing in astronomical images using deep learning techniques. The project is implemented in PyTorch and includes a variety of model architectures, optimization strategies, and evaluation metrics.

## Project Overview

Gravitational lenses are rare astronomical phenomena where the gravitational field of a massive object (like a galaxy) bends the light from a distant object behind it, creating distorted or multiple images. This project's goal is to build a binary classifier to distinguish between images containing gravitational lenses and those without.

Key challenges include:
- Class imbalance (non-lenses are much more common than lenses)
- Subtle visual features that distinguish lenses
- Need for high precision while maintaining good recall

## Results

The project achieved excellent performance with an ensemble approach combining ResNet50 and EfficientNet models, reaching:
- **ROC AUC: 0.9932**
- **F1 Score: 0.7455**
- **Accuracy: 0.9943**

### Model Performance Comparison

| Model Configuration | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------| 
| Ensemble (ResNet50+EfficientNet+EfficientNet ModifiedReLU) | 0.9943 | 0.6694 | 0.8410 | 0.7455 | 0.9932 |
| EfficientNet B0 + Modified ReLU | 0.9944 | 0.6856 | 0.8051 | 0.7406 | 0.9877 |
| ResNet50 + Heavy Augmentation | 0.9937 | 0.6449 | 0.8103 | 0.7182 | 0.9914 |
| EfficientNet B0 + Modified ReLU (50 epochs) | 0.9929 | 0.5972 | 0.8667 | 0.7071 | 0.9909 |
| EfficientNet B0 + Heavy Augmentation | 0.9919 | 0.5612 | 0.8462 | 0.6748 | 0.9897 |

See `docs/usage_guide.md` for complete results and analysis.

## Key Features

- **Multiple model architectures**: ResNet18/50, EfficientNet B0, DenseNet121, custom CNN
- **Advanced techniques**:
  - Physics-guided attention mechanisms
  - FFT-augmented feature extraction
  - Modified activation functions
  - Ensemble methods
- **Specialized loss functions**: Focal Loss, Tversky Loss, Physics-Aware Loss
- **Comprehensive evaluation**: ROC-AUC curves, precision-recall metrics, confusion matrices

## Repository Structure

```
deeplens-classification/
├── config.py               # Configuration parameters
├── data_utils.py           # Data loading and processing utilities
├── models.py               # Model architectures
├── train_utils.py          # Training functions
├── eval_utils.py           # Evaluation metrics and visualization
├── loss_functions.py       # Custom loss functions
├── model_io.py             # Model saving and loading
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── data/                   # Data directory
│   ├── train_lenses/       # Training lens images
│   ├── train_nonlenses/    # Training non-lens images
│   ├── test_lenses/        # Test lens images
│   └── test_nonlenses/     # Test non-lens images
├── saved_models/           # Directory for saved models
├── notebooks/              # Jupyter notebooks
│  ├── deeplense_lense_finding.ipynb    # Data exploration notebook
│  └── ensemble.ipynb       # Ensembling notebook
├── docs/                   # Documentation
│  └── usage_guide.md       # Usage documentation
├── rocs_curves/            # some roc curves
│  └──images...
└── common_task/        
   ├── readme.md
   ├── best_model.pth
   ├── deeplense_common_task.ipynb
   └── roc_curves/
      └──images...
```

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deeplens-classification.git
cd deeplens-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the data:
   - Download the dataset and place in the `data/` directory
   - Ensure the structure matches the expected format (see Repository Structure)

### Basic Usage

Train a model:
```bash
python main.py --mode train --model resnet50 --batch_size 32 --epochs 50
```

Evaluate a trained model:
```bash
python main.py --mode evaluate --model_path saved_models/resnet50_20250324-120000_model.pth
```

Use ensemble of models:
```bash
python main.py --mode evaluate --ensemble --model_paths saved_models/model1.pth saved_models/model2.pth --metadata_paths saved_models/model1_metadata.pth saved_models/model2_metadata.pth
```

For more detailed usage instructions, see `docs/usage_guide.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided for the DeepLense gravitational lens classification challenge
- PyTorch and torchvision teams for the deep learning framework