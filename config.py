import torch

# Configuration parameters for the lens classification model
config = {
   'batch_size': 32,
   'num_epochs': 50,
   'learning_rate': 0.001,
   'val_split': 0.2,
   'seed': 42,
   'model_name': 'resnet50',  # Options: 'resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', 'custom_cnn', 'efficient_attention', 'efficient_fft_attention', 'fftaugmented', 'crop_view', 'dinov2-variants'
   'freeze_backbone': False,
   'optimizer': 'adam',  # Options: 'adam', 'sgd', 'adamw'
   'scheduler': 'cosine',  # Options: 'step', 'cosine', 'none'
   'early_stopping_patience': 10,
   'weight_decay': 1e-4,
   'dropout_rate': 0,
   'img_size': 64,
   'save_path': './saved_models',
   'num_workers': 4,
   'augmentation_level': 'medium',  # Options: 'none', 'light', 'medium', 'heavy'
   'rotation': 360,
   'mean': [0.485, 0.456, 0.406],
   'std': [0.229, 0.224, 0.225],
   'use_attention': False,
   'loss_function': 'bce',  # Options: 'bce', 'pin', 'focal', 'tversky', 'focaltversky'
   'focal_alpha': (0.25, 0.75),
   'focal_gamma': 3.5,
   'tversky_alpha_beta': (0.3, 0.7),
   'focaltversky_a_b_g': (0.3, 0.7, 0.2),
   'einstien_check': False,
   'modified_relu': False,
   'weighted_sampling': False,
   'data_path': './data'  # Path to the data folder
}

# Ensemble configuration
ensemble_config = {
    'weights': [1, 1, 1],  # Default equal weights
    'use_weights': False
}

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")