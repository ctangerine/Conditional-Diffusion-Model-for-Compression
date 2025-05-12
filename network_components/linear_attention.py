import torch
import torch.nn as nn
from einops import rearrange

class LinearAttention(nn.Module):
    '''
    Description:
        Linear Attention, a self-attention mechanism that uses linear transformation to calculate the query, key, and value.
        The attention mechanism is calculated by the dot product of the query and key, and the output is calculated by the dot product of the context and query.
    Args:
        in_channel: int, the number of input channels
        heads: int, the number of heads
        head_dimention: int, the number of channels in each head
        device: torch.device, device to use for computation (default: None, will use CUDA if available)
    Operations:
        1. Apply a 1x1 convolutional layer to the input tensor to calculate the query, key, and value
        2. Split the query, key, and value tensor into multiple heads
        3. Calculate the query tensor by scaling the query tensor
        4. Calculate the key tensor by applying the softmax function to the key tensor
        5. Calculate the context tensor by the dot product of the key and value tensor
        6. Calculate the output tensor by the dot product of the context and query tensor
    '''

    def __init__(self, in_channel, heads=1, head_dimention=None, device=None):
        super().__init__()
        # Set device based on availability if not provided
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if (head_dimention is None):
            head_dimention = in_channel 
        self.heads = heads
        self.head_dimention = head_dimention
        self.in_channel = in_channel
        self.scale = head_dimention ** -0.5
        self.hidden_dimention = heads * head_dimention
        self.to_QueryKeyValue = nn.Conv2d(in_channels=in_channel, out_channels=self.hidden_dimention * 3, bias=False, kernel_size=1)
        self.to_output = nn.Conv2d(in_channels=self.hidden_dimention, out_channels=in_channel, kernel_size=1)
        
        # Move model components to device
        self.to(self.device)

    def forward(self, input):
        # Move input to device
        input = input.to(self.device)
        
        batch_size, channel, height, width = input.size()
        query_key_value = self.to_QueryKeyValue(input)
        query_key_value = query_key_value.chunk(3, dim=1)
        query, key, value = map(lambda x: rearrange(x, "b (h c) x y -> b h c (x y)", h=self.heads), query_key_value)
        query = query * self.scale

        key = key.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", key, value)

        output = torch.einsum("b h d e, b h d n -> b h e n", context, query)
        output = rearrange(output, "b h c (x y) -> b (h c) x y", h=self.heads, x=height, y=width)

        return self.to_output(output)