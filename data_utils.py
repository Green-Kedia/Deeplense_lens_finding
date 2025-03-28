import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import random
from config import config, device

def set_seed(seed=42):
   """Set seeds for reproducibility"""
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

class LensDataset(Dataset):
   def __init__(self, file_paths, labels, transform=None):
       """
       Args:
           file_paths (list): List of paths to .npy files
           labels (list): List of labels corresponding to file_paths
           transform (callable, optional): Optional transform to be applied on a sample
       """
       self.file_paths = file_paths
       self.labels = labels
       self.transform = transform
       
   def __len__(self):
       return len(self.file_paths)
   
   def __getitem__(self, idx):
       # Load the npy file
       image = np.load(self.file_paths[idx])
       
       # Convert to PyTorch tensor (C, H, W)
       image = torch.from_numpy(image).float()
       
       # Apply transformations if provided
       if self.transform:
           image = self.transform(image)
       
       return image, self.labels[idx]

def prepare_data(train_lenses, train_nonlenses, config):
   """
   Prepare the datasets and dataloaders
   
   Args:
       train_lenses (list): List of file paths to lens data
       train_nonlenses (list): List of file paths to non-lens data
       config (dict): Configuration parameters
       
   Returns:
       train_loader, val_loader: DataLoader objects for training and validation
   """
   set_seed(config['seed'])
   
   # Create file paths and labels lists
   all_file_paths = train_lenses + train_nonlenses
   all_labels = [1] * len(train_lenses) + [0] * len(train_nonlenses)
   
   # Define augmentations based on the selected level
   if config['augmentation_level'] == 'none':
       transform = transforms.Compose([
           transforms.Normalize(mean=config['mean'], std=config['std'])
       ])
   elif config['augmentation_level'] == 'light':
       transform = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.Normalize(mean=config['mean'], std=config['std'])
       ])
   elif config['augmentation_level'] == 'medium':
       transform = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.RandomVerticalFlip(),
           transforms.RandomRotation(config['rotation']),
           transforms.Normalize(mean=config['mean'], std=config['std'])
       ])
   else:  # heavy
       transform = transforms.Compose([
           transforms.RandomHorizontalFlip(),
           transforms.RandomVerticalFlip(),
           transforms.RandomRotation(config['rotation']),
           transforms.ColorJitter(brightness=0.1, contrast=0.1),
           transforms.Normalize(mean=config['mean'], std=config['std'])
       ])
   
   # Create the full dataset
   if config['model_name'].startswith('dinov2'):
       full_dataset = LensDataset(all_file_paths, all_labels, transform=None)
   else:
       full_dataset = LensDataset(all_file_paths, all_labels, transform=transform)
   
   # Split into train and validation sets
   val_size = int(config['val_split'] * len(full_dataset))
   train_size = len(full_dataset) - val_size
   
   train_dataset, val_dataset = random_split(
       full_dataset, [train_size, val_size], 
       generator=torch.Generator().manual_seed(config['seed'])
   )
   
   # Create data loaders
   if config['model_name'].startswith('dinov2'):
       train_loader = DataLoader(
           train_dataset, 
           batch_size=config['batch_size'], 
           shuffle=False, 
           num_workers=config['num_workers'],
           pin_memory=True
       )
       
   elif config['weighted_sampling']:
       labels = [full_dataset.labels[i] for i in train_dataset.indices]
       class_counts = np.bincount(labels)  # Count occurrences of 0s and 1s
       class_weights = 1.0 / class_counts  # Inverse of frequency
       sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float)
       
       # Create sampler
       sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset.indices), replacement=True)
       train_loader = DataLoader(
           train_dataset, 
           batch_size=config['batch_size'],
           sampler=sampler,
           num_workers=config['num_workers'],
           pin_memory=True
       )
   else:
       train_loader = DataLoader(
           train_dataset, 
           batch_size=config['batch_size'], 
           shuffle=True, 
           num_workers=config['num_workers'],
           pin_memory=True
       )
   
   val_loader = DataLoader(
       val_dataset, 
       batch_size=config['batch_size'], 
       shuffle=False, 
       num_workers=config['num_workers'],
       pin_memory=True
   )
   
   print(f"Training samples: {len(train_dataset)}")
   print(f"Validation samples: {len(val_dataset)}")
   
   return train_loader, val_loader

def prepare_test_data(test_lenses, test_nonlenses, config):
   """
   Prepare the test dataset
   
   Args:
       test_lenses (list): List of file paths to test lens data
       test_nonlenses (list): List of file paths to test non-lens data
       config (dict): Configuration parameters
       
   Returns:
       test_loader: DataLoader for test data
   """
   # Create file paths and labels lists
   test_file_paths = test_lenses + test_nonlenses
   test_labels = [1] * len(test_lenses) + [0] * len(test_nonlenses)
   
   # Use a simple normalization transform for test data (no augmentation)
   transform = transforms.Compose([
       transforms.Normalize(mean=config['mean'], std=config['std'])
   ])
   
   # Create the test dataset
   if config['model_name'].startswith('dinov2'):
       test_dataset = LensDataset(test_file_paths, test_labels, transform=None)
   else:
       test_dataset = LensDataset(test_file_paths, test_labels, transform=transform)
   
   # Create data loader
   test_loader = DataLoader(
       test_dataset, 
       batch_size=config['batch_size'], 
       shuffle=False, 
       num_workers=config['num_workers'],
       pin_memory=True
   )
   
   print(f"Test samples: {len(test_dataset)}")
   print(f"Lens samples: {len(test_lenses)}")
   print(f"Non-lens samples: {len(test_nonlenses)}")
   
   return test_loader

def visualize_data(data_loader, num_images=5):
   """
   Visualize samples from the dataset
   
   Args:
       data_loader: DataLoader object to visualize samples from
       num_images (int): Number of images to visualize
   """
   # Get a batch of data
   images, labels = next(iter(data_loader))
   
   # Create a figure to display the images
   fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
   
   # Display the images
   for i in range(num_images):
       if i < len(images):
           # Convert the tensor to a numpy array and transpose to (H, W, C) for display
           img = images[i].numpy()
           
           # Normalize the image for visualization
           img = (img - img.min()) / (img.max() - img.min())
           
           # Display the image
           axes[i].imshow(np.transpose(img, (1, 2, 0)))
           axes[i].set_title(f"Class: {'Lens' if labels[i] == 1 else 'Non-Lens'}")
           axes[i].axis('off')
   
   plt.tight_layout()
   plt.show()
   
   # Plot class distribution
   all_labels = []
   for _, batch_labels in data_loader:
       all_labels.extend(batch_labels.numpy())
   
   # Count labels
   unique_labels, counts = np.unique(all_labels, return_counts=True)
   
   # Plot class distribution
   plt.figure(figsize=(8, 5))
   plt.bar(['Non-Lens', 'Lens'], [counts[0], counts[1]], color=['skyblue', 'salmon'])
   plt.title('Class Distribution')
   plt.ylabel('Count')
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   plt.show()

def get_data_loaders():
   """
   Load data from the data paths defined in config and prepare data loaders
   
   Returns:
       train_loader, val_loader
   """
   # Get file paths
   data_path = config['data_path']
   train_lenses = []
   train_nonlenses = []
   test_lenses = []
   test_nonlenses = []
   
   # Training data
   lens_dir = os.path.join(data_path, 'train_lenses')
   for filename in os.listdir(lens_dir):
       if filename.endswith('.npy'):
           train_lenses.append(os.path.join(lens_dir, filename))
   
   nonlens_dir = os.path.join(data_path, 'train_nonlenses')
   for filename in os.listdir(nonlens_dir):
       if filename.endswith('.npy'):
           train_nonlenses.append(os.path.join(nonlens_dir, filename))
   
   # Test data
   lens_dir = os.path.join(data_path, 'test_lenses')
   for filename in os.listdir(lens_dir):
       if filename.endswith('.npy'):
           test_lenses.append(os.path.join(lens_dir, filename))
   
   nonlens_dir = os.path.join(data_path, 'test_nonlenses')
   for filename in os.listdir(nonlens_dir):
       if filename.endswith('.npy'):
           test_nonlenses.append(os.path.join(nonlens_dir, filename))
   
   # Create data loaders
   train_loader, val_loader = prepare_data(train_lenses, train_nonlenses, config)
   test_loader = prepare_test_data(test_lenses, test_nonlenses, config)
   
   return train_loader, val_loader, test_loader