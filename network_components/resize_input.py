import torch
import torch.nn as nn

class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel, device=None):
        super().__init__()
        # Set device based on availability if not provided
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        
        # Move model components to device
        self.to(self.device)

    def forward(self, input):
        # Move input to device
        # input = input.to(self.device)
        return self.conv_layer(input)
    
class UpSampling(nn.Module):
    def __init__(self, in_channel, out_channel, device=None):
        super().__init__()
        # Set device based on availability if not provided
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.conv_layer = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1)
        
        # Move model components to device
        self.to(self.device)

    def forward(self, input):
        # Move input to device
        input = input.to(self.device)
        output = self.conv_layer(input)
        return output