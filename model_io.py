import os
import time
import torch
from config import config, device

def save_model(model, config, metrics, history, filename=None):
   """
   Save the trained model and related data
   
   Args:
       model (nn.Module): The model to save
       config (dict): Configuration parameters
       metrics (dict): Evaluation metrics
       history (dict): Training history
       filename (str, optional): Specific filename to use
   """
   # Create save directory if it doesn't exist
   os.makedirs(config['save_path'], exist_ok=True)
   
   # Generate filename if not provided
   if filename is None:
       timestamp = time.strftime("%Y%m%d-%H%M%S")
       filename = f"{config['model_name']}_{timestamp}"
   
   # Save model state dict
   model_path = os.path.join(config['save_path'], f"{filename}_model.pth")
   if isinstance(model, tuple) and len(model) == 2:
       # For DINOv2 models, save only the classifier part
       torch.save(model[1].state_dict(), model_path)
   else:
       torch.save(model.state_dict(), model_path)
   
   # Save config, metrics, and history
   metadata = {
       'config': config,
       'metrics': metrics,
       'history': history
   }
   metadata_path = os.path.join(config['save_path'], f"{filename}_metadata.pth")
   torch.save(metadata, metadata_path)
   
   print(f"Model saved to {model_path}")
   print(f"Metadata saved to {metadata_path}")

def load_model(model_path, metadata_path=None):
   """
   Load a saved model and related data
   
   Args:
       model_path (str): Path to the saved model
       metadata_path (str, optional): Path to the saved metadata
       
   Returns:
       model (nn.Module): The loaded model
       metadata (dict, optional): The loaded metadata if metadata_path is provided
   """
   from models import create_model
   
   # Load metadata if path is provided
   metadata = None
   if metadata_path is not None:
       metadata = torch.load(metadata_path)
       config = metadata['config']
       
       # Create model with the saved configuration
       model = create_model(config)
   else:
       # Create a default model if no metadata is provided
       from torchvision import models
       import torch.nn as nn
       
       model = models.resnet18(weights=None)
       model.fc = nn.Sequential(
           nn.Dropout(0.2),
           nn.Linear(model.fc.in_features, 1),
           nn.Sigmoid()
       )
       model = model.to(device)
   
   # Load model weights
   if isinstance(model, tuple) and len(model) == 2:
       # For DINOv2 models, load only the classifier part
       model[1].load_state_dict(torch.load(model_path))
   else:
       model.load_state_dict(torch.load(model_path))
   
   print(f"Model loaded from {model_path}")
   if metadata_path is not None:
       print(f"Metadata loaded from {metadata_path}")
   
   return model, metadata