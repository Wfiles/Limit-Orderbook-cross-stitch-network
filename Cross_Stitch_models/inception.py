import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, input_channels):
        super(InceptionModule, self).__init__()
        
        # First branch: Conv -> Conv
        self.branch1_1x1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 1), padding='same')
        self.branch1_3x1 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same')
        
        # Second branch: Conv -> Conv
        self.branch2_1x1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 1), padding='same')
        self.branch2_5x1 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same')
        
        # Third branch: MaxPool -> Conv
        self.branch3_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.branch3_1x1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 1), padding='same')
        
        # LeakyReLU Activation
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # First branch
        branch1 = self.activation(self.branch1_1x1(x))
        branch1 = self.activation(self.branch1_3x1(branch1))
        
        # Second branch
        branch2 = self.activation(self.branch2_1x1(x))
        branch2 = self.activation(self.branch2_5x1(branch2))
        
        # Third branch
        branch3 = self.branch3_pool(x)
        branch3 = self.activation(self.branch3_1x1(branch3))
        
        # Concatenate branches along channel dimension
        outputs = torch.cat([branch1, branch2, branch3], dim=1).squeeze(3)  # Concatenate along channels (dim=1)       
        output_drop = self.dropout(outputs)
        return output_drop
  
