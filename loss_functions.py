import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class FocalLoss(nn.Module):
   def __init__(self, alpha=(0.5, 0.5), gamma=2.0, reduction='mean'):
       """
       alpha: Tuple (alpha_neg, alpha_pos) for class balancing
       gamma: Focusing parameter (higher -> more focus on hard examples)
       reduction: 'mean' or 'sum'
       """
       super(FocalLoss, self).__init__()
       self.alpha = torch.tensor(alpha)  # Convert tuple to tensor
       self.gamma = gamma
       self.reduction = reduction

   def forward(self, inputs, targets):
       BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
       pt = torch.exp(-BCE_loss)  # Probability of correct class
       alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)  # Select alpha for each sample
       
       focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

       if self.reduction == 'mean':
           return focal_loss.mean()
       elif self.reduction == 'sum':
           return focal_loss.sum()
       return focal_loss  # No reduction

class TverskyLoss(nn.Module):
   def __init__(self):
       super(TverskyLoss, self).__init__()
       self.alpha = config['tversky_alpha_beta'][0]  # Penalizes false negatives
       self.beta = config['tversky_alpha_beta'][1]    # Penalizes false positives (increase for higher precision)

   def forward(self, inputs, targets, smooth=1e-6):
       inputs = inputs.view(-1)  # Flatten
       targets = targets.view(-1)  # Flatten
       
       # True Positive, False Positive, False Negative
       TP = (inputs * targets).sum()
       FP = ((1 - targets) * inputs).sum()
       FN = (targets * (1 - inputs)).sum()
       
       tversky_index = (TP + smooth) / (TP + self.alpha * FN + self.beta * FP + smooth)
       loss = 1 - tversky_index
       
       return loss

class FocalTverskyLoss(nn.Module):
   def __init__(self):
       super(FocalTverskyLoss, self).__init__()
       self.alpha = config['focaltversky_a_b_g'][0]
       self.beta = config['focaltversky_a_b_g'][1]
       self.gamma = config['focaltversky_a_b_g'][2]

   def forward(self, inputs, targets, smooth=1e-6):
       inputs = inputs.view(-1)  # Flatten
       targets = targets.view(-1)  # Flatten
       
       # True Positive, False Positive, False Negative
       TP = (inputs * targets).sum()
       FP = ((1 - targets) * inputs).sum()
       FN = (targets * (1 - inputs)).sum()
       
       tversky_index = (TP + smooth) / (TP + self.alpha * FN + self.beta * FP + smooth)
       focal_tversky_loss = (1 - tversky_index) ** self.gamma
       
       return focal_tversky_loss

class PhysicsAwareLoss(nn.Module):
   def __init__(self, alpha=0.5):
       super().__init__()
       self.focal_loss = nn.BCELoss()  # Handle class imbalance
       self.alpha = alpha  # Weight for radius penalty
       
   def forward(self, outputs, targets):
       cls_logits, radius_pred = outputs
       cls_loss = self.focal_loss(cls_logits, targets)
       
       # Radius constraint (valid range: 0.5-5 arcseconds)
       radius_penalty = torch.mean(
           torch.clamp(radius_pred - 5, min=0) + 
           torch.clamp(0.5 - radius_pred, min=0)
       )
       
       total_loss = cls_loss + self.alpha * radius_penalty
       return total_loss

def erc(r):
   """Einstein Ring Check - verifies if the predicted radius is in a valid range"""
   if config['einstien_check']:
       return (r >= 0.5) & (r <= 5)
   else:
       return r == r  # Always true, effectively disables the check