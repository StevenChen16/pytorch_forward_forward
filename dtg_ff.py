import torch
import torch.nn as nn
import torch.nn.functional as F

class DTG_Layer(nn.Module):
    """Enhanced Dynamic Temperature Goodness Layer
    
    Improvements:
    1. Learnable feature clarity calculation
    2. Temperature history tracking and stabilization
    3. Improved feature normalization
    """
    def __init__(self, in_dim, out_dim, temp_min=0.1, temp_max=2.0, history_size=100):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        
        # Feature clarity learning
        self.clarity_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Temperature parameters
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.threshold = nn.Parameter(torch.zeros(1))
        
        # Temperature history
        self.register_buffer('temp_history', torch.zeros(history_size))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def calc_feature_clarity(self, z):
        """Calculate learnable feature clarity"""
        z_mean = z.mean(dim=0)
        z_std = z.std(dim=0)
        
        # Construct feature clarity input
        clarity_input = torch.stack([
            (z_std / (torch.abs(z_mean) + 1e-6)).mean(),
            (z_mean.abs() / (z_std + 1e-6)).mean()
        ])
        
        # Learn feature clarity
        feature_clarity = self.clarity_fc(clarity_input)
        return feature_clarity.squeeze()
    
    def update_temp_history(self, temp):
        """Update temperature history"""
        idx = self.history_ptr.item()
        self.temp_history[idx] = temp
        self.history_ptr[0] = (idx + 1) % self.temp_history.size(0)
    
    def get_stable_temp(self, current_temp):
        """Get stabilized temperature value"""
        valid_temps = self.temp_history[self.temp_history > 0]
        if len(valid_temps) > 0:
            history_mean = valid_temps.mean()
            history_std = valid_temps.std()
            
            # Use history statistics to limit temperature changes
            max_change = 0.1 * history_mean
            stable_temp = torch.clamp(
                current_temp,
                min=history_mean - max_change,
                max=history_mean + max_change
            )
            
            # Additional stability constraints
            if history_std > 0.1:
                # If temperature fluctuates significantly, prefer historical mean
                alpha = torch.sigmoid(history_std * 10)
                stable_temp = alpha * history_mean + (1 - alpha) * stable_temp
            
            return stable_temp
        return current_temp
    
    def forward(self, x):
        """Forward propagation"""
        # Feature transformation
        z = self.bn(self.linear(x))
        z = F.relu(z)
        
        if self.training:
            # Calculate feature clarity and temperature
            clarity = self.calc_feature_clarity(z)
            raw_temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(clarity)
            temp = self.get_stable_temp(raw_temp)
            
            # Update temperature history
            with torch.no_grad():
                self.update_temp_history(temp.detach())
            
            # Calculate goodness
            z_norm = F.normalize(z, p=2, dim=1)
            goodness = torch.sum((z_norm / (temp + 1e-6)) ** 2, dim=1)
            threshold = F.softplus(self.threshold)
            
            return {
                "features": z,
                "goodness": goodness,
                "temperature": temp,
                "threshold": threshold,
                "clarity": clarity
            }
        else:
            return z

class DTG_Model(nn.Module):
    """Enhanced Dynamic Temperature Goodness Model"""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # Network structure
        self.layers = nn.ModuleList()
        for i in range(opt.model.num_layers):
            in_dim = opt.input_dim if i == 0 else opt.model.hidden_dim
            self.layers.append(DTG_Layer(
                in_dim=in_dim,
                out_dim=opt.model.hidden_dim,
                temp_min=0.1,
                temp_max=2.0
            ))
        
        # Classifier
        self.classifier = nn.Linear(opt.model.hidden_dim, opt.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def calc_loss(self, goodness, labels, temp, threshold, clarity=None):
        """Calculate loss
        
        Added clarity regularization term
        """
        # Dynamic margin
        margin = torch.clamp(threshold * temp, min=0.1, max=2.0)
        
        # Process positive and negative samples separately
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        # Main loss
        pos_loss = F.relu(threshold + margin - goodness[pos_mask]).mean()
        neg_loss = F.relu(goodness[neg_mask] - (threshold - margin)).mean()
        
        # Add clarity regularization
        if clarity is not None:
            # Encourage moderate clarity values
            clarity_reg = 0.1 * (clarity ** 2).mean()
        else:
            clarity_reg = 0
        
        return pos_loss + neg_loss + clarity_reg
    
    def forward(self, inputs, labels=None):
        """Forward propagation"""
        outputs = {
            "Loss": 0.0,
            "Temperature": 0.0,
            "Clarity": 0.0
        }
        
        if not isinstance(inputs, dict):
            return outputs
        
        # Process positive samples
        pos_z = inputs["pos_images"].view(inputs["pos_images"].shape[0], -1)
        z = pos_z
        for i, layer in enumerate(self.layers):
            if self.training:
                layer_out = layer(z)
                z = layer_out["features"]
                
                # Calculate loss
                pos_labels = torch.ones_like(layer_out["goodness"])
                loss = self.calc_loss(
                    layer_out["goodness"],
                    pos_labels,
                    layer_out["temperature"],
                    layer_out["threshold"],
                    layer_out["clarity"]
                )
                outputs["Loss"] += loss
                outputs["Temperature"] += layer_out["temperature"]
                outputs["Clarity"] += layer_out["clarity"]
                
                # Record accuracy
                outputs[f"ff_accuracy_layer_{i}"] = (
                    layer_out["goodness"] > layer_out["threshold"]
                ).float().mean()
            else:
                z = layer(z)
        
        # Process negative samples
        neg_z = inputs["neg_images"].view(inputs["neg_images"].shape[0], -1)
        z = neg_z
        for i, layer in enumerate(self.layers):
            if self.training:
                layer_out = layer(z)
                z = layer_out["features"]
                
                # Calculate loss
                neg_labels = torch.zeros_like(layer_out["goodness"])
                loss = self.calc_loss(
                    layer_out["goodness"],
                    neg_labels,
                    layer_out["temperature"],
                    layer_out["threshold"],
                    layer_out["clarity"]
                )
                outputs["Loss"] += loss
            else:
                z = layer(z)
        
        # Classification task
        if "neutral_sample" in inputs and labels is not None and "class_labels" in labels:
            neutral_z = inputs["neutral_sample"].view(inputs["neutral_sample"].shape[0], -1)
            z = neutral_z
            
            # Extract features
            for layer in self.layers[:-1]:
                if self.training:
                    z = layer(z)["features"]
                else:
                    z = layer(z)
            
            # Classification prediction
            logits = self.classifier(z)
            cls_loss = F.cross_entropy(logits, labels["class_labels"])
            outputs["Loss"] += cls_loss
            
            # Calculate classification accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                outputs["classification_accuracy"] = (
                    preds == labels["class_labels"]
                ).float().mean()
        
        # Calculate averages
        num_layers = len(self.layers)
        outputs["Temperature"] /= num_layers
        outputs["Clarity"] /= num_layers
        
        return outputs

class DTG_Config:
    """Enhanced Dynamic Temperature Goodness Configuration"""
    def __init__(self):
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Input settings
        self.input = type('', (), {})()
        self.input.path = "data"
        self.input.batch_size = 128
        self.input.dataset = "mnist"
        
        # Model settings
        self.model = type('', (), {})()
        self.model.hidden_dim = 2048
        self.model.num_layers = 4
        
        # Training settings
        self.training = type('', (), {})()
        self.training.epochs = 100
        self.training.learning_rate = 1e-3
        self.training.weight_decay = 1e-4
        self.training.momentum = 0.9
        
        # Set input dimensions based on dataset
        self.input_dim = 784  # Default for MNIST
        self.num_classes = 10