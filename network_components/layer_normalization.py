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
    Operations:
        1. Calculate the variance and mean of the input tensor by channels (dim=1), other cases: dim=0 (batch), dim=2 (height), dim=3 (width)
        2. Normalize the input tensor by the variance and mean
        3. Scale and shift the normalized tensor by gamma and beta
    '''

    def __init__(self, in_dimension, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, in_dimension, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1,in_dimension, 1, 1))

    def forward(self, input):
        var = torch.var(input, dim=1, keepdim=True, unbiased=False)
        mean = torch.mean(input, dim=1, keepdim=True)
        return self.gamma * (input - mean) / torch.sqrt(var + self.eps) + self.beta
        