import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score, 
   confusion_matrix, roc_curve, roc_auc_score
)

from config import config, device, ensemble_config
from data_utils import set_seed
from loss_functions import erc

def evaluate_model(model, data_loader):
   """
   Evaluate the model on the given data loader
   
   Args:
       model (nn.Module): The model to evaluate
       data_loader (DataLoader): DataLoader for evaluation data
       
   Returns:
       metrics (dict): Evaluation metrics
   """
   model.eval()
   all_preds = []
   all_scores = []  # Added to store raw prediction scores
   all_labels = []
   
   # Collect predictions and labels
   with torch.no_grad():
       for inputs, labels in tqdm(data_loader, desc="Evaluating"):
           inputs = inputs.to(device)
           
           # Forward pass
           outputs = model(inputs)
           preds = ((outputs[0] > 0.5) & erc(outputs[1])).float() if type(outputs)==tuple else (outputs > 0.5).float()
           
           # Collect predictions, raw scores, and labels
           all_preds.extend(preds.cpu().numpy())
           all_scores.extend(outputs[0].cpu().numpy() if type(outputs)==tuple else outputs.cpu().numpy())  # Save raw scores for ROC AUC
           all_labels.extend(labels.numpy())
   
   # Convert to numpy arrays
   all_preds = np.array(all_preds).flatten()
   all_scores = np.array(all_scores).flatten()
   all_labels = np.array(all_labels)
   
   # Calculate metrics
   acc = accuracy_score(all_labels, all_preds)
   precision = precision_score(all_labels, all_preds, zero_division=0)
   recall = recall_score(all_labels, all_preds, zero_division=0)
   f1 = f1_score(all_labels, all_preds, zero_division=0)
   cm = confusion_matrix(all_labels, all_preds)
   
   # Calculate ROC AUC
   roc_auc = roc_auc_score(all_labels, all_scores)
   
   # Print results
   print(f'Accuracy: {acc:.4f}')
   print(f'Precision: {precision:.4f}')
   print(f'Recall: {recall:.4f}')
   print(f'F1 Score: {f1:.4f}')
   print(f'ROC AUC: {roc_auc:.4f}')
   
   # Create figure with subplots
   plt.figure(figsize=(15, 6))
   
   # Plot confusion matrix
   plt.subplot(1, 2, 1)
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Lens', 'Lens'],
               yticklabels=['Non-Lens', 'Lens'])
   plt.xlabel('Predicted')
   plt.ylabel('Actual')
   plt.title('Confusion Matrix')
   
   # Plot ROC curve
   plt.subplot(1, 2, 2)
   fpr, tpr, _ = roc_curve(all_labels, all_scores)
   plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic')
   plt.legend(loc="lower right")
   
   plt.tight_layout()
   plt.show()
   
   # Return metrics
   metrics = {
       'accuracy': acc,
       'precision': precision,
       'recall': recall,
       'f1': f1,
       'roc_auc': roc_auc,  # Added ROC AUC to metrics
       'confusion_matrix': cm
   }
   
   return metrics

def evaluate_model_with_dinov2(dinov2_model, classifier_model, data_loader):
   """
   Evaluate the model using DINOv2 embeddings on the given data loader
   
   Args:
       dinov2_model (nn.Module): The DINOv2 model for extracting embeddings
       classifier_model (nn.Module): The classifier model to evaluate
       data_loader (DataLoader): DataLoader for evaluation data
       
   Returns:
       metrics (dict): Evaluation metrics
   """
   # Set seed for reproducibility
   set_seed(config['seed'])
   
   print("Extracting DINOv2 embeddings for evaluation...")
   # Extract embeddings from DINOv2 model
   dinov2_model.eval()
   all_embeddings = []
   all_labels = []
   
   # Extract embeddings
   with torch.no_grad():
       for inputs, labels in tqdm(data_loader, desc="Extracting embeddings"):
           inputs = inputs.to(device)
           # Get embeddings from DINOv2
           embeddings = dinov2_model.get_embeddings(inputs)
           
           all_embeddings.append(embeddings.cpu())
           all_labels.append(labels)
   
   embeddings_tensor = torch.cat(all_embeddings)
   labels_tensor = torch.cat(all_labels)
   
   # Create a new dataset and dataloader with the embeddings
   embedding_dataset = TensorDataset(embeddings_tensor, labels_tensor)
   embedding_loader = DataLoader(
       embedding_dataset,
       batch_size=data_loader.batch_size,
       shuffle=False,
       num_workers=data_loader.num_workers if hasattr(data_loader, 'num_workers') else 0
   )
   
   # Use the standard evaluation function with the embedding data loader
   return evaluate_model(classifier_model, embedding_loader)

def plot_training_history(history):
   """
   Plot the training history
   
   Args:
       history (dict): Training history
   """
   # Plot training and validation loss
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 2, 1)
   plt.plot(history['train_loss'], label='Training Loss')
   plt.plot(history['val_loss'], label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training and Validation Loss')
   plt.legend()
   plt.grid(linestyle='--', alpha=0.6)
   
   # Plot training and validation accuracy
   plt.subplot(1, 2, 2)
   plt.plot(history['train_acc'], label='Training Accuracy')
   plt.plot(history['val_acc'], label='Validation Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.title('Training and Validation Accuracy')
   plt.legend()
   plt.grid(linestyle='--', alpha=0.6)
   
   plt.tight_layout()
   plt.show()
   
   # Plot validation metrics
   plt.figure(figsize=(16, 4))
   
   plt.subplot(1, 4, 1)
   plt.plot(history['val_precision'], marker='o')
   plt.xlabel('Epoch')
   plt.ylabel('Precision')
   plt.title('Validation Precision')
   plt.grid(linestyle='--', alpha=0.6)
   
   plt.subplot(1, 4, 2)
   plt.plot(history['val_recall'], marker='o', color='green')
   plt.xlabel('Epoch')
   plt.ylabel('Recall')
   plt.title('Validation Recall')
   plt.grid(linestyle='--', alpha=0.6)
   
   plt.subplot(1, 4, 3)
   plt.plot(history['val_f1'], marker='o', color='purple')
   plt.xlabel('Epoch')
   plt.ylabel('F1 Score')
   plt.title('Validation F1 Score')
   plt.grid(linestyle='--', alpha=0.6)
   
   # Add ROC AUC plot
   plt.subplot(1, 4, 4)
   plt.plot(history['val_roc_auc'], marker='o', color='orange')
   plt.xlabel('Epoch')
   plt.ylabel('ROC AUC')
   plt.title('Validation ROC AUC')
   plt.grid(linestyle='--', alpha=0.6)
   
   plt.tight_layout()
   plt.show()

def evaluate_on_test_set(model, test_loader):
   """
   Evaluate a trained model on the test set
   
   Args:
       model: Trained PyTorch model
       test_loader: DataLoader for test data
       
   Returns:
       test_metrics: Dictionary of test metrics
   """
   print("Evaluating model on test set...")
   
   # Evaluate on test data
   if isinstance(model, tuple) and len(model) == 2:
       test_metrics = evaluate_model_with_dinov2(model[0], model[1], test_loader)
   else:
       test_metrics = evaluate_model(model, test_loader)
   
   # Print summary
   print("\nTest Set Evaluation Summary:")
   print(f"Accuracy: {test_metrics['accuracy']:.4f}")
   print(f"Precision: {test_metrics['precision']:.4f}")
   print(f"Recall: {test_metrics['recall']:.4f}")
   print(f"F1 Score: {test_metrics['f1']:.4f}")
   print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
   
   return test_metrics