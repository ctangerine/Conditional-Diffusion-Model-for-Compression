import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''
    Description:
        Layer Normalization, define a way to normalize the input tensor by channels. Instead of using fixed variance and mean,
        it use nn.Parameter (which will be learned during training) to calculate the variance and mean.
    Args:
        in_dimension: int, the number of input channels
        eps: float, a small number to avoid dividing by zero
        device: torch.device, device to use for computation (default: None, will use CUDA if available)
    Operations:
        1. Calculate the variance and mean of the input tensor by channels (dim=1), other cases: dim=0 (batch), dim=2 (height), dim=3 (width)
        2. Normalize the input tensor by the variance and mean
        3. Scale and shift the normalized tensor by gamma and beta
    '''

    def __init__(self, in_dimension, eps=1e-5, device=None):
        super().__init__()
        # Set device based on availability if not provided
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, in_dimension, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_dimension, 1, 1))
        
        # Move parameters to device
        self.to(self.device)

    def forward(self, input):
        # Move input to device
        input = input.to(self.device)
        
        var = torch.var(input, dim=1, keepdim=True, unbiased=False)
        mean = torch.mean(input, dim=1, keepdim=True)
        return self.gamma * (input - mean) / torch.sqrt(var + self.eps) + self.beta
        

class BaseLayerNorm(nn.Module):
    def __init__(self, functional, dimension, device=None):
        super().__init__()
        # Set device based on availability if not provided
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.functional = functional
        self.norm = LayerNorm(dimension, eps=1e-5, device=self.device)
        
        # Move components to device
        self.to(self.device)
        
    def forward(self, input):
        # Move input to device
        input = input.to(self.device)
        
        input = self.norm(input)
        return self.functional(input)