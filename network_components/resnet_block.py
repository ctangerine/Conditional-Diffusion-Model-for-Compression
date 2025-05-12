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

    def __init__(self, in_channel, out_channel, large_kernel=False, time_embedding=False, time_embedding_channels=None):
        super().__init__()
        self.kernel_size = 5 if large_kernel else 3
        # Set device based on CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if time_embedding is not False:
            self.mlp = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True), nn.Linear(time_embedding_channels, out_channel),
            )
        else:
            self.mlp = None
        
        self.resnet_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            LayerNorm(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LayerNorm(out_channel),
            nn.ReLU(),
        )

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        # Move model components to device
        self.to(self.device)

    def forward(self, input, time_tensor=None): 
        # Move input to the same device as the model
        input = input.to(self.device)
        
        if time_tensor is not None:
            time_tensor = time_tensor.to(self.device)
            
        self.time_tensor = time_tensor if time_tensor is not None else None

        conv = None
        for module in self.resnet_block:
            conv = module(input) if conv is None else module(conv)

        if self.mlp is not None:
            time_emb = self.mlp(time_tensor.clone())
            conv = conv + time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)

        shortcut = self.shortcut(input)
        return conv + shortcut
    

class BaseResidualBlock(nn.Module):
    def __init__(self, functional):
        super(BaseResidualBlock, self).__init__()
        # Set device based on CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.functional = functional
        self.layer_norm = LayerNorm(functional.in_channels)
        
        # Move model components to device
        self.to(self.device)

    def forward(self, x):
        # Move input to the same device as the model
        x = x.to(self.device)
        return x + self.functional(x)
