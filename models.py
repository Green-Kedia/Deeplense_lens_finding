import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoImageProcessor
from config import config, device, ensemble_config

class ModifiedReLU(nn.Module):
   def forward(self, x):
       return (1 / torch.sqrt(torch.tensor(torch.pi - 1))) * (torch.sqrt(2 * torch.tensor(torch.pi)) * torch.relu(x) - 1)

def replace_activation(model, new_activation):
   """Replace all ReLU and SiLU activations in a model with a custom activation"""
   for name, module in model.named_children():
       if isinstance(module, (nn.SiLU, nn.ReLU)):  
           setattr(model, name, new_activation)
       else:
           replace_activation(module, new_activation)
   return model

class DINOv2Embedder(nn.Module):
   """
   DINOv2 model for feature extraction
   """
   def __init__(self, config):
       super().__init__()
       self.processor = AutoImageProcessor.from_pretrained(f"facebook/{config['model_name']}")
       self.model = AutoModel.from_pretrained(f"facebook/{config['model_name']}")
       
       # Freeze the backbone if specified
       for param in self.model.parameters():
           param.requires_grad = False
               
       # Store the embedding dimension for downstream tasks
       self.embedding_dim = self.model.config.hidden_size
   
   def forward(self, x):
       """
       Forward pass without classification head - just returns embeddings
       """
       inputs = self.processor(images=x, return_tensors="pt").to(x.device)
       outputs = self.model(**inputs)
       # Extract CLS token embedding
       embeddings = outputs.last_hidden_state[:, 0, :]
       return embeddings
   
   def get_embeddings(self, x):
       """
       Explicit method to get embeddings (same as forward, but more explicit)
       """
       return self.forward(x)

class EmbeddingClassifier(nn.Module):
   """
   Classifier model that works with DINOv2 embeddings
   """
   def __init__(self, config, embedding_dim):
       super().__init__()
       
       # Simple classifier with optional multi-head output
       self.use_multi_head = config.get('use_multi_head', False)
       self.dropout_rate = config.get('dropout_rate', 0.2)
       
       if self.use_multi_head:
           # Main classification head
           self.main_head = nn.Sequential(
               nn.Dropout(self.dropout_rate),
               nn.Linear(embedding_dim, 1),
               nn.Sigmoid()
           )
           
           # Auxiliary head for Einstein Ring Classification (erc)
           self.aux_head = nn.Sequential(
               nn.Dropout(self.dropout_rate),
               nn.Linear(embedding_dim, 1),
               nn.Sigmoid()
           )
       else:
           # Single classification head
           self.fc = nn.Sequential(
               nn.Dropout(self.dropout_rate),
               nn.Linear(embedding_dim, 1),
               nn.Sigmoid()
           )
   
   def forward(self, x):
       """
       Forward pass that takes embeddings and returns classification
       """
       if self.use_multi_head:
           main_output = self.main_head(x)
           aux_output = self.aux_head(x)
           return (main_output, aux_output)
       else:
           return self.fc(x)

class efficient_attention(nn.Module):
   def __init__(self, num_classes=1, use_attention=True):
       super().__init__()
       # Backbone: EfficientNet-B0
       self.backbone = models.efficientnet_b0(pretrained=True).features
       self.use_attention = use_attention
       
       # Physics-Guided Attention
       if self.use_attention:
           self.radial_mask = self._create_radial_mask(64)  # For input size 64x64
           self.channel_att = nn.Sequential(
               nn.AdaptiveAvgPool2d(1),
               nn.Conv2d(3, 16, 1), nn.ReLU(),
               nn.Conv2d(16, 3, 1), nn.Sigmoid()
           )
       
       # Classification & Regression Heads
       self.classifier = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Flatten(),
           nn.Dropout(config['dropout_rate']),
           nn.Linear(1280, num_classes),
           nn.Sigmoid()
       )
       self.radius_head = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Flatten(),
           nn.Dropout(config['dropout_rate']),
           nn.Linear(1280, 1)
       )

   def _create_radial_mask(self, size):
       """Fixed Gaussian mask to focus on central regions"""
       xx, yy = torch.meshgrid(torch.arange(size), torch.arange(size))
       mask = torch.exp(-((xx - size//2)**2 + (yy - size//2)**2)/(2*(size//4)**2))
       return mask.unsqueeze(0)  # Shape: (1, H, W)

   def forward(self, x):
       # Apply attention
       if self.use_attention:
           x = x * self.radial_mask.to(x.device)  # Radial bias
           channel_weights = self.channel_att(x)
           x = x * channel_weights
       
       # Backbone features
       features = self.backbone(x)
       
       # Outputs
       cls_logits = self.classifier(features)
       radius = self.radius_head(features)
       return cls_logits, radius

class FFTAugmentedModel(nn.Module):
   def __init__(self):
       super().__init__()
       # Spatial branch (original RGB)
       self.spatial_branch = models.efficientnet_b0(pretrained=True).features
       
       # Frequency branch (FFT magnitudes)
       self.freq_branch = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2)
       )
       
       # Combined classifier
       self.classifier = nn.Sequential(
           nn.Dropout(config['dropout_rate']),
           nn.Linear(4*1280 + 32*32*32, 1),
           nn.Sigmoid()
       )  # Adjust dimensions

   def forward(self, x):
       # Spatial features
       spatial_feat = self.spatial_branch(x)
       
       # Frequency features
       fft = torch.fft.fft2(x, dim=(-2, -1))
       fft_mag = torch.log(1 + torch.abs(fft))  # Log-scale magnitudes
       freq_feat = self.freq_branch(fft_mag)
       
       # Concatenate & classify
       combined = torch.cat([spatial_feat.flatten(1), 
                            freq_feat.flatten(1)], dim=1)
       
       return self.classifier(combined)

class efficient_fft_attention(nn.Module):
   def __init__(self, num_classes=1):
       super().__init__()
       # Backbone: EfficientNet-B0
       self.backbone = models.efficientnet_b0(pretrained=True).features
       
       # Physics-Guided Attention
       self.channel_att = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Conv2d(3, 16, 1), nn.ReLU(),
           nn.Conv2d(16, 3, 1), nn.Sigmoid()
       )
       
       # Classification & Regression Heads
       self.classifier = nn.Sequential(
           nn.AdaptiveAvgPool2d(1),
           nn.Flatten(),
           nn.Dropout(config['dropout_rate']),
           nn.Linear(1280, num_classes),
           nn.Sigmoid()
       )

   def forward(self, x):
       # Apply attention
       fft = torch.fft.fft2(x, dim=(-2, -1))
       fft_mag = torch.log(1 + torch.abs(fft))
       channel_weights = self.channel_att(fft_mag)
       x = x * channel_weights
       
       # Backbone features
       features = self.backbone(x)
       
       # Outputs
       cls_logits = self.classifier(features)
       return cls_logits

class crop_view(nn.Module):
   def __init__(self):
       super().__init__()
       # Full-image branch (64x64 input)
       self.full_net = self._build_effnet()
       
       # Center-crop branch (32x32 input)
       self.crop_net = self._build_effnet()
       
       # Classifier (combines both branches)
       self.classifier = nn.Sequential(
           nn.Linear(2560, 512),  # 1280 (full) + 1280 (crop) = 2560
           nn.ReLU(),
           nn.Dropout(config['dropout_rate']),
           nn.Linear(512, 1),
           nn.Sigmoid()
       )

   def _build_effnet(self):
       """EfficientNet-B0 without classifier + adjusted for small inputs"""
       model = models.efficientnet_b0(pretrained=False)
       model.classifier = nn.Identity()  # Remove final FC layer
       
       # Modify stem convolution for smaller inputs
       model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
       return model

   def forward(self, x):
       # Full-image branch
       full_feat = self.full_net(x)
       
       # Center crop (32x32 from 64x64 input)
       _, _, H, W = x.shape
       x_crop = x[:, :, H//4:3*H//4, W//4:3*W//4]  # Center 32x32
       
       # Crop branch
       crop_feat = self.crop_net(x_crop)
       
       # Combine and classify
       combined = torch.cat([full_feat, crop_feat], dim=1)
       return self.classifier(combined)
   
class ensemble(nn.Module):
    def __init__(self, paths):
        super().__init__()
        models = []
        for path, model_path in paths:
            metadata = torch.load(path)
            config = metadata['config']
            
            # Create model with the saved configuration
            model = create_model(config)
            model.load_state_dict(torch.load(model_path))
            models.append(model)
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(ensemble_config['weights'], dtype=torch.float32, device=device).view(-1, 1, 1) 
        
    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models], dim=0)  # Shape: (num_models, batch_size, 1)
        if ensemble_config['use_weights']:
            weighted_outputs = outputs * self.weights  # Element-wise multiplication
            avg_output = weighted_outputs.sum(dim=0)  # Averaging probabilities
        else:
            avg_output = outputs.mean(dim=0)
        return avg_output

def create_model(config):
   """
   Create a model based on the configuration
   
   Args:
       config (dict): Configuration parameters
       
   Returns:
       model (nn.Module): The created model
   """
   if config['model_name'] == 'resnet18':
       model = models.resnet18(weights='DEFAULT')
       if config['freeze_backbone']:
           for param in model.parameters():
               param.requires_grad = False
       # Modify the final layer for binary classification
       model.fc = nn.Sequential(
           nn.Dropout(config['dropout_rate']),
           nn.Linear(model.fc.in_features, 1),
           nn.Sigmoid()
       )

   elif config['model_name'].startswith('dinov2'):
       dinov2_model = DINOv2Embedder(config)
       classifier_model = EmbeddingClassifier(
           config, 
           embedding_dim=dinov2_model.embedding_dim
       )
       model = (dinov2_model, classifier_model)

   elif config['model_name'] == 'efficient_attention':
       model = efficient_attention(use_attention = config['use_attention'])

   elif config['model_name'] == 'fftaugmented':
       model = FFTAugmentedModel()

   elif config['model_name'] == 'efficient_fft_attention':
       model = efficient_fft_attention()

   elif config['model_name'] == 'crop_view':
       model = crop_view()
   
   elif config['model_name'] == 'resnet50':
       model = models.resnet50(weights='DEFAULT')
       if config['freeze_backbone']:
           for param in model.parameters():
               param.requires_grad = False
       # Modify the final layer
       model.fc = nn.Sequential(
           nn.Dropout(config['dropout_rate']),
           nn.Linear(model.fc.in_features, 1),
           nn.Sigmoid()
       )
   
   elif config['model_name'] == 'efficientnet_b0':
       model = models.efficientnet_b0(weights='DEFAULT')
       if config['freeze_backbone']:
           for param in model.parameters():
               param.requires_grad = False
       # Modify the classifier
       model.classifier = nn.Sequential(
           nn.Dropout(config['dropout_rate']),
           nn.Linear(model.classifier[1].in_features, 1),
           nn.Sigmoid()
       )
   
   elif config['model_name'] == 'densenet121':
       model = models.densenet121(weights='DEFAULT')
       if config['freeze_backbone']:
           for param in model.parameters():
               param.requires_grad = False
       # Modify the classifier
       model.classifier = nn.Sequential(
           nn.Dropout(config['dropout_rate']),
           nn.Linear(model.classifier.in_features, 1),
           nn.Sigmoid()
       )
   
   elif config['model_name'] == 'custom_cnn':
       # Define a custom CNN architecture
       model = nn.Sequential(
           # Input shape: (3, 64, 64)
           nn.Conv2d(3, 32, kernel_size=3, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 32, 32)
           
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 16, 16)
           
           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.BatchNorm2d(128),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 8, 8)
           
           nn.Conv2d(128, 256, kernel_size=3, padding=1),
           nn.BatchNorm2d(256),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 4, 4)
           
           nn.Flatten(),
           nn.Linear(256 * 4 * 4, 512),
           nn.ReLU(),
           nn.Dropout(config['dropout_rate']),
           nn.Linear(512, 1),
           nn.Sigmoid()
       )
   
   else:
       raise ValueError(f"Model {config['model_name']} not supported")

   if config['modified_relu']:
       model = (model[0], replace_activation(model[1], ModifiedReLU())) if type(model)==tuple else replace_activation(model, ModifiedReLU())
   
   # Move model to device
   model = (model[0].to(device), model[1].to(device)) if type(model)==tuple else model.to(device)
   print(f"Created {config['model_name']} model")
   
   return model