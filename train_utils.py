import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import config, device, ensemble_config
from data_utils import set_seed
from loss_functions import FocalLoss, TverskyLoss, FocalTverskyLoss, PhysicsAwareLoss, erc

def get_optimizer_and_scheduler(model, config):
   """
   Create an optimizer and learning rate scheduler based on the configuration
   
   Args:
       model (nn.Module): The model to optimize
       config (dict): Configuration parameters
       
   Returns:
       optimizer, scheduler: The created optimizer and scheduler
   """
   # Create optimizer
   if config['optimizer'] == 'adam':
       optimizer = optim.Adam(
           model.parameters(), 
           lr=config['learning_rate'],
           weight_decay=config['weight_decay']
       )
   elif config['optimizer'] == 'sgd':
       optimizer = optim.SGD(
           model.parameters(), 
           lr=config['learning_rate'],
           momentum=0.9,
           weight_decay=config['weight_decay']
       )
   elif config['optimizer'] == 'adamw':
       optimizer = optim.AdamW(
           model.parameters(),
           lr=config['learning_rate'],
           weight_decay=config['weight_decay']
       )
   else:
       raise ValueError(f"Optimizer {config['optimizer']} not supported")
   
   # Create scheduler
   if config['scheduler'] == 'step':
       scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
   elif config['scheduler'] == 'cosine':
       scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
   elif config['scheduler'] == 'none':
       scheduler = None
   else:
       raise ValueError(f"Scheduler {config['scheduler']} not supported")
   
   return optimizer, scheduler

def train_model(model, train_loader, val_loader, config):
   """
   Train the model
   
   Args:
       model (nn.Module): The model to train
       train_loader (DataLoader): DataLoader for training data
       val_loader (DataLoader): DataLoader for validation data
       config (dict): Configuration parameters
       
   Returns:
       model (nn.Module): The trained model
       history (dict): Training history
   """
   set_seed(config['seed'])
   
   # Create optimizer and scheduler
   optimizer, scheduler = get_optimizer_and_scheduler(model, config)
   
   # Define loss function
   if config['loss_function'] == 'bce':
       criterion = nn.BCELoss()
       print('Using bce loss')

   elif config['loss_function'] == 'focal':
       alpha = config['focal_alpha'] # (~0.057, ~0.943)
       gamma = config['focal_gamma']  # Standard choice
       
       criterion = FocalLoss(alpha=alpha, gamma=gamma)

       print(f'Using focal loss with alpha {alpha} and gamma {gamma}')

   elif config['loss_function'] == 'tversky':
       
       criterion = TverskyLoss()

       print(f"Using tversky loss with a_b {config['tversky_alpha_beta']}")

   elif config['loss_function'] == 'focaltversky':
       
       criterion = FocalTverskyLoss()

       print(f"Using focaltversky loss with a_b_g {config['focaltversky_a_b_g']}")

   elif config['loss_function'] == 'pin':
       criterion = PhysicsAwareLoss()
       print('Using pin loss')
   
   # Initialize the best model weights and validation accuracy
   best_model_wts = copy.deepcopy(model.state_dict())
   best_val_auc = 0.0
   best_epoch = 0
   patience_counter = 0
   
   # Initialize history
   history = {
       'train_loss': [],
       'val_loss': [],
       'train_acc': [],
       'val_acc': [],
       'val_precision': [],
       'val_recall': [],
       'val_f1': [],
       'val_roc_auc': []  # Added ROC AUC metric
   }
   
   # Training loop
   for epoch in range(config['num_epochs']):
       print(f'Epoch {epoch+1}/{config["num_epochs"]}')
       print('-' * 10)
       
       # Training phase
       model.train()
       running_loss = 0.0
       running_corrects = 0
       
       # Iterate over the training data
       for inputs, labels in tqdm(train_loader, desc="Training"):
           inputs = inputs.to(device)
           labels = labels.float().to(device).view(-1, 1)
           
           # Zero the parameter gradients
           optimizer.zero_grad()
           
           # Forward pass
           with torch.set_grad_enabled(True):
               outputs = model(inputs)
               loss = criterion(outputs, labels)
               
               # Backward pass + optimize
               loss.backward()
               optimizer.step()
           
           # Statistics
           running_loss += loss.item() * inputs.size(0)
           preds = ((outputs[0] > 0.5) & erc(outputs[1])).float() if type(outputs)==tuple else (outputs > 0.5).float()
           running_corrects += torch.sum(preds == labels)
       
       if scheduler is not None:
           scheduler.step()
       
       # Calculate epoch statistics
       epoch_loss = running_loss / len(train_loader.dataset)
       epoch_acc = running_corrects.double() / len(train_loader.dataset)
       
       print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
       
       # Validation phase
       model.eval()
       val_running_loss = 0.0
       val_preds = []
       val_labels = []
       val_scores = []  # Added for ROC AUC calculation
       
       # Iterate over the validation data
       for inputs, labels in tqdm(val_loader, desc="Validation"):
           inputs = inputs.to(device)
           labels = labels.float().to(device).view(-1, 1)
           
           # Forward pass
           with torch.no_grad():
               outputs = model(inputs)
               loss = criterion(outputs, labels)
           
           # Statistics
           val_running_loss += loss.item() * inputs.size(0)
           preds = ((outputs[0] > 0.5) & erc(outputs[1])).float() if type(outputs)==tuple else (outputs > 0.5).float()
           
           # Collect predictions, raw scores, and labels for metrics
           val_preds.extend(preds.cpu().numpy())
           val_scores.extend(outputs[0].cpu().numpy() if type(outputs)==tuple else outputs.cpu().numpy())  # Raw scores for ROC AUC
           val_labels.extend(labels.cpu().numpy())
       
       # Calculate validation statistics
       val_epoch_loss = val_running_loss / len(val_loader.dataset)
       val_preds = np.array(val_preds).flatten()
       val_scores = np.array(val_scores).flatten()  # Convert to numpy array
       val_labels = np.array(val_labels).flatten()
       
       val_acc = accuracy_score(val_labels, val_preds)
       val_precision = precision_score(val_labels, val_preds, zero_division=0)
       val_recall = recall_score(val_labels, val_preds, zero_division=0)
       val_f1 = f1_score(val_labels, val_preds, zero_division=0)
       
       # Calculate ROC AUC
       val_roc_auc = roc_auc_score(val_labels, val_scores)
       
       print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_acc:.4f} Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f} ROC AUC: {val_roc_auc:.4f}')
       
       # Update history
       history['train_loss'].append(epoch_loss)
       history['val_loss'].append(val_epoch_loss)
       history['train_acc'].append(epoch_acc.item())
       history['val_acc'].append(val_acc)
       history['val_precision'].append(val_precision)
       history['val_recall'].append(val_recall)
       history['val_f1'].append(val_f1)
       history['val_roc_auc'].append(val_roc_auc)  # Add ROC AUC to history
       
       # If validation accuracy improved, save the model weights
       if val_roc_auc > best_val_auc:
           best_val_auc = val_roc_auc
           best_model_wts = copy.deepcopy(model.state_dict())
           best_epoch = epoch
           patience_counter = 0
       else:
           patience_counter += 1
           
       # Early stopping
       if patience_counter >= config['early_stopping_patience']:
           print(f'Early stopping triggered after epoch {epoch+1}')
           break
           
       print()
   
   # Load best model weights
   model.load_state_dict(best_model_wts)
   print(f'Best validation auc: {best_val_auc:.4f} achieved at epoch {best_epoch+1}')
   
   return model, history

def extract_dinov2_embeddings(model, data_loader):
   """
   Extract DINOv2 embeddings for all samples in the data loader
   
   Args:
       model (nn.Module): The DINOv2 model
       data_loader (DataLoader): DataLoader containing the data
       
   Returns:
       embeddings (torch.Tensor): Extracted embeddings
       labels (torch.Tensor): Corresponding labels
   """
   model.eval()
   all_embeddings = []
   all_labels = []
   
   with torch.no_grad():
       for inputs, labels in tqdm(data_loader, desc="Extracting DINOv2 embeddings"):
           inputs = inputs.to(device)
           # Get embeddings from DINOv2
           embeddings = model.get_embeddings(inputs)
           
           all_embeddings.append(embeddings.cpu())
           all_labels.append(labels)
   
   return torch.cat(all_embeddings), torch.cat(all_labels)

def train_model_with_dinov2(dinov2_model, classifier_model, train_loader, val_loader, config):
   """
   Train the model using pre-computed DINOv2 embeddings
   
   Args:
       dinov2_model (nn.Module): The DINOv2 model for extracting embeddings
       classifier_model (nn.Module): The classifier model to train
       train_loader (DataLoader): DataLoader for training data
       val_loader (DataLoader): DataLoader for validation data
       config (dict): Configuration parameters
       
   Returns:
       model (nn.Module): The trained classifier model
       history (dict): Training history
   """
   # Set seed for reproducibility
   set_seed(config['seed'])
   
   print("Extracting DINOv2 embeddings...")
   # Extract DINOv2 embeddings for training and validation sets
   train_embeddings, train_labels = extract_dinov2_embeddings(dinov2_model, train_loader)
   val_embeddings, val_labels = extract_dinov2_embeddings(dinov2_model, val_loader)
   
   # Create new data loaders with the extracted embeddings
   train_dataset = TensorDataset(train_embeddings, train_labels)
   val_dataset = TensorDataset(val_embeddings, val_labels)
   
   if config.get('weighted_sampling', False):
       # Convert labels to numpy for counting
       labels_np = train_labels.numpy().flatten()
       class_counts = np.bincount(labels_np.astype(int))  # Count occurrences of 0s and 1s
       class_weights = 1.0 / class_counts  # Inverse of frequency
       sample_weights = torch.tensor([class_weights[int(label)] for label in labels_np], dtype=torch.float)
       
       # Create sampler
       sampler = torch.utils.data.WeightedRandomSampler(
           sample_weights, 
           num_samples=len(train_labels), 
           replacement=True
       )
       
       train_emb_loader = DataLoader(
           train_dataset,
           batch_size=train_loader.batch_size,
           sampler=sampler,  # Use sampler instead of shuffle
           num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0,
           pin_memory=True
       )
   else:
       # Standard DataLoader with shuffling if no weighted sampling
       train_emb_loader = DataLoader(
           train_dataset,
           batch_size=train_loader.batch_size,
           shuffle=True,
           num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0,
           pin_memory=True
       )
   
   val_emb_loader = DataLoader(
       val_dataset,
       batch_size=val_loader.batch_size,
       shuffle=False,
       num_workers=val_loader.num_workers if hasattr(val_loader, 'num_workers') else 0,
       pin_memory=True
   )
   
   # Call the standard training function with the embedding loaders
   # Note: For DINOv2 we only train the classifier head, not the embedder
   return train_model(classifier_model, train_emb_loader, val_emb_loader, config)

def run_full_pipeline(data_dict, config):
   """
   Run the full training pipeline
   
   Args:
       data_dict (dict): Dictionary with 'train_lenses' and 'train_nonlenses' keys
       config (dict): Configuration parameters
   """
   from models import create_model
   from data_utils import prepare_data, visualize_data
   from eval_utils import evaluate_model, plot_training_history, evaluate_model_with_dinov2
   from model_io import save_model
   
   # Prepare data
   train_loader, val_loader = prepare_data(
       data_dict['train_lenses'], 
       data_dict['train_nonlenses'], 
       config
   )
   
   # Visualize some samples
   visualize_data(train_loader)
   
   # Create model
   model = create_model(config)
   
   # Train model
   if config['model_name'].startswith('dinov2'):
       trained_model, history = train_model_with_dinov2(model[0], model[1], train_loader, val_loader, config)
       model = (model[0], trained_model)  # Replace the classifier component while keeping the embedder
   else:
       trained_model, history = train_model(model, train_loader, val_loader, config)
   
   # Evaluate model
   if config['model_name'].startswith('dinov2'):
       metrics = evaluate_model_with_dinov2(model[0], model[1], val_loader)
   else:
       metrics = evaluate_model(trained_model, val_loader)
   
   # Plot training history
   plot_training_history(history)
   
   # Save model
   timestamp = time.strftime("%Y%m%d-%H%M%S")
   save_model(trained_model, config, metrics, history, 
              filename=f"{config['model_name']}_{timestamp}")
   
   return trained_model, metrics, history