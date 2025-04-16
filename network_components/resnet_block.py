import torch
import torch.nn as nn

from network_components.layer_normalization import LayerNorm 

class ResnetBlock(nn.Module):
    '''
    Description:
        Residual block with Layer Normalization, the block contains two convolutional layers with Layer Normalization and ReLU activation.
        The block also contains a shortcut connection to skip the block if the input and output channels are the same.  
    Args:
        in_channel: int, the number of input channels
        out_channel: int, the number of output channels
        large_kernel: bool, if True, the kernel size of the first convolutional layer will be 5, otherwise 3
    Operations:
        1. Apply the first convolutional layer with Layer Normalization and ReLU activation
        2. Apply the second convolutional layer with Layer Normalization and ReLU activation
        3. Apply the shortcut connection to skip the block if the input and output channels are the same
    '''

    def __init__(self, in_channel, out_channel, large_kernel=False):
        super().__init__()
        self.kernel_size = 5 if large_kernel else 3

        self.resnes_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=self.kernel_size),
            LayerNorm(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3),
            LayerNorm(out_channel),
            nn.ReLU(),
        )

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, input): 
        return self.resnes_block(input) + self.shortcut(input)