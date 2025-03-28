# Gravitational Lens Classification

## Project Overview
This project implements a deep learning model for the classification of strong gravitational lensing images into three categories:
1. No substructure
2. Subhalo substructure
3. Vortex substructure

## Model Architecture
The model uses EfficientNet-B0 as the backbone, which provides excellent performance while maintaining computational efficiency. This pre-trained network is fine-tuned on the gravitational lensing dataset.

## Implementation Details
- **Framework**: PyTorch
- **Model**: EfficientNet-B0 with modified classification head
- **Data Handling**: Custom dataset class for gravitational lens images
- **Training Strategy**: Transfer learning with a pre-trained ImageNet model

## Data Preprocessing
- Images are loaded from .npy files and converted to a suitable format for the model
- Images are resized to 224x224 pixels
- Normalization is applied to improve training stability
- A 90/10 train/validation split is used to monitor performance during training

## Results
- The model achieves high classification accuracy across all three classes
- ROC curves and AUC scores demonstrate excellent discrimination capability
- Class-specific AUC scores:
  - Class 0 (no substructure): 0.9861
  - Class 1 (subhalo): 0.9744
  - Class 2 (vortex): 0.9872

## Usage
To run the notebook:
1. Ensure PyTorch and required dependencies are installed
2. Download the dataset from the provided link
3. Update the dataset path in the notebook if necessary
4. Execute the notebook cells sequentially

## Files
- `deeplense-common-task.ipynb`: Jupyter notebook containing the full implementation
- `best-model.pth`: Model weights
- `README.md`: This file

## Requirements
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- tqdm