import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import modules
from config import config, device, ensemble_config
from data_utils import get_data_loaders, visualize_data, set_seed
from models import create_model, ensemble
from train_utils import train_model, train_model_with_dinov2, run_full_pipeline
from eval_utils import evaluate_model, evaluate_model_with_dinov2, evaluate_on_test_set, plot_training_history
from model_io import save_model, load_model

def parse_args():
   parser = argparse.ArgumentParser(description='Gravitational Lens Classification')
   parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'],
                       help='Operation mode: train, evaluate, or visualize data')
   parser.add_argument('--model', type=str, default=config['model_name'],
                       help='Model architecture (default: resnet50)')
   parser.add_argument('--batch_size', type=int, default=config['batch_size'],
                       help='Batch size for training (default: 32)')
   parser.add_argument('--epochs', type=int, default=config['num_epochs'],
                       help='Number of epochs to train (default: 50)')
   parser.add_argument('--lr', type=float, default=config['learning_rate'],
                       help='Learning rate (default: 0.001)')
   parser.add_argument('--data_path', type=str, default=config['data_path'],
                       help='Path to the data directory')
   parser.add_argument('--save_path', type=str, default=config['save_path'],
                       help='Path to save model and results')
   parser.add_argument('--model_path', type=str, default=None,
                       help='Path to load pretrained model for evaluation')
   parser.add_argument('--metadata_path', type=str, default=None,
                       help='Path to load model metadata for evaluation')
   parser.add_argument('--seed', type=int, default=config['seed'],
                       help='Random seed for reproducibility')
   parser.add_argument('--loss', type=str, default=config['loss_function'],
                       choices=['bce', 'focal', 'tversky', 'focaltversky', 'pin'],
                       help='Loss function to use')
   parser.add_argument('--optimizer', type=str, default=config['optimizer'],
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer to use')
   parser.add_argument('--augmentation', type=str, default=config['augmentation_level'],
                       choices=['none', 'light', 'medium', 'heavy'],
                       help='Data augmentation level')
   parser.add_argument('--weighted_sampling', action='store_true',
                       help='Use weighted sampling to handle class imbalance')
   parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble of models for evaluation only')
   parser.add_argument('--model_paths', nargs='+', type=str, default=[],
                       help='List of model paths to use in ensemble')
   parser.add_argument('--metadata_paths', nargs='+', type=str, default=[],
                       help='List of metadata paths matching the models')
   parser.add_argument('--ensemble_weights', nargs='+', type=float, default=[],
                       help='Weights for ensemble models')
   
   return parser.parse_args()

def update_config(args):
   """Update config with command line arguments"""
   config['model_name'] = args.model
   config['batch_size'] = args.batch_size
   config['num_epochs'] = args.epochs
   config['learning_rate'] = args.lr
   config['data_path'] = args.data_path
   config['save_path'] = args.save_path
   config['seed'] = args.seed
   config['loss_function'] = args.loss
   config['optimizer'] = args.optimizer
   config['augmentation_level'] = args.augmentation
   config['weighted_sampling'] = args.weighted_sampling
   
   return config

def main():
   # Parse command-line arguments
   args = parse_args()
   
   # Update configuration with command-line arguments
   updated_config = update_config(args)
   
   # Set random seed for reproducibility
   set_seed(updated_config['seed'])
   
   # Print system information
   print(f"Using device: {device}")
   print(f"PyTorch version: {torch.__version__}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   
   # Print configuration
   print("\nConfiguration:")
   for key, value in updated_config.items():
       print(f"  {key}: {value}")
   
   # Get data loaders
   train_loader, val_loader, test_loader = get_data_loaders()

   if args.ensemble:
       if len(args.model_paths) != len(args.metadata_paths):
            print("Error: Number of model paths must match number of metadata paths")
            return
            
        # Setup ensemble weights if provided
       if args.ensemble_weights:
            from config import ensemble_config
            ensemble_config['weights'] = args.ensemble_weights
            ensemble_config['use_weights'] = True
            
        # Create ensemble model
       model_pairs = list(zip(args.metadata_paths, args.model_paths))
       ensemble_model = ensemble(model_pairs)

       # Evaluate ensemble on test set
       print("\nEvaluating ensemble on test set:")
       test_metrics = evaluate_on_test_set(ensemble_model, test_loader)
   
   elif args.mode == 'visualize':
       # Visualize data
       print("\nVisualizing training data samples:")
       visualize_data(train_loader)
       
       # Visualize test data
       print("\nVisualizing test data samples:")
       visualize_data(test_loader)
       
   elif args.mode == 'train':
       # Create model
       model = create_model(updated_config)
       
       # Train model
       print("\nStarting model training...")
       if updated_config['model_name'].startswith('dinov2'):
           trained_model, history = train_model_with_dinov2(model[0], model[1], train_loader, val_loader, updated_config)
           model = (model[0], trained_model)  # Replace the classifier component
       else:
           trained_model, history = train_model(model, train_loader, val_loader, updated_config)
       
       # Plot training history
       plot_training_history(history)
       
       # Evaluate on validation set
       print("\nEvaluating on validation set:")
       if updated_config['model_name'].startswith('dinov2'):
           val_metrics = evaluate_model_with_dinov2(model[0], model[1], val_loader)
       else:
           val_metrics = evaluate_model(trained_model, val_loader)
       
       # Evaluate on test set
       print("\nEvaluating on test set:")
       test_metrics = evaluate_on_test_set(model if updated_config['model_name'].startswith('dinov2') else trained_model, test_loader)
       
       # Save model
       save_model(model if updated_config['model_name'].startswith('dinov2') else trained_model, 
                 updated_config, test_metrics, history)
       
   elif args.mode == 'evaluate':
       if args.model_path is None:
           print("Error: Model path is required for evaluation mode")
           return
       
       # Load model
       model, metadata = load_model(args.model_path, args.metadata_path)
       
       # Evaluate on test set
       test_metrics = evaluate_on_test_set(model, test_loader)
       
       # Print test metrics
       print("\nTest Set Metrics:")
       for key, value in test_metrics.items():
           if key != 'confusion_matrix':
               print(f"  {key}: {value:.4f}")
   
   print("\nDone!")

if __name__ == "__main__":
   main()