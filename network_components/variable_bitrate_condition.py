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
    Operations:
        1. Apply a 1x1 convolutional layer to the condition tensor to calculate the scale and shift
        2. Scale and shift the input tensor by the scale and shift
    '''

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.scale = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, bias=False)
        self.shift = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, bias=False)

    def forward(self, input, condition):
        condition = condition.reshape(-1, 1, 1, 1)
        scale = self.scale(condition)
        shift = self.shift(condition)
        return input * scale + shift