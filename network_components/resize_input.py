import torch
import torch.nn as nn

class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        return self.conv_layer(input)
    
class UpSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, input):
        return self.conv_layer(input)