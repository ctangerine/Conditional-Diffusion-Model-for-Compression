import torch
import torch.nn as nn

class VariableBitrateCondition(nn.Module):
    '''
    Description:
        Variable Bitrate Condition, a module that uses the condition to calculate the scale and shift of the input tensor.
        The scale and shift are calculated by applying a 1x1 convolutional layer to the condition tensor.
    Args:
        input_channel: int, the number of input channels
        output_channel: int, the number of output channels
        device: torch.device, device to use for computation (default: None, will use CUDA if available)
    Operations:
        1. Apply a 1x1 convolutional layer to the condition tensor to calculate the scale and shift
        2. Scale and shift the input tensor by the scale and shift
    '''

    def __init__(self, input_channel, output_channel, device=None):
        super().__init__()
        # Set device based on availability if not provided
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.scale = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, bias=False)
        self.shift = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, bias=False)
        
        # Move model components to device
        self.to(self.device)

    def forward(self, input, condition):
        # Move inputs to device
        input = input.to(self.device)
        condition = condition.to(self.device)
        
        condition = condition.reshape(-1, 1, 1, 1)
        scale = self.scale(condition)
        shift = self.shift(condition)
        return input * scale + shift