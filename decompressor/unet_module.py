import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path so Python can find the network_components module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LOGGER import setup_logger
logger = setup_logger('unet_module', 'unet_module.log', level='WARNING')

from network_components.resnet_block import ResnetBlock
from network_components.resize_input import *
from network_components.layer_normalization import *
from network_components.linear_attention import *
from network_components.utils import *

import torchsummary


class UnetModule(nn.Module):
    def __init__(self, 
        base_channels=3, 
        channels_multiplier=[1, 2, 4, 8, 16, 32, 64],
        input_channels=3,
        output_channels=3,
        context_channels=[3, 64, 128, 192],
        time_embedding=True,
    ):
        super(UnetModule, self).__init__()
        self.base_channels = base_channels
        self.channels_muliplier = channels_multiplier 
        self.output_channels =  output_channels
        self.input_channels = input_channels
        self.context_channels = context_channels
        self.time_embedding = time_embedding 

        self.layer_channels = [int(base_channels * multiplier) for multiplier in channels_multiplier]
        if self.context_channels is None:
            self.context_channels = self.layer_channels

        self.encoder_channels_pair = list(zip(self.layer_channels[:-1], self.layer_channels[1:]))
        self.decoder_channels_pair = list(zip(self.layer_channels[::-1][:-1], self.layer_channels[::-1][1:]))
        logger.info(f"Encoder channels pair: {self.encoder_channels_pair}")
        logger.info(f"Decoder channels pair: {self.decoder_channels_pair}")
        logger.info(f"Layer channels: {self.layer_channels}")
        logger.info(f"Context channels: {self.context_channels}")

        if self.time_embedding:
            self.time_embedding_layer = nn.Sequential(
                nn.Linear(1, self.base_channels*4),
                nn.GELU(),
                nn.Linear(self.base_channels*4, self.base_channels),
            )

        else:
            self.time_embedding_layer = None
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottleneck = nn.ModuleList()


        self.build_network()

    
    def build_network(self):
        num_layers = len(self.encoder_channels_pair)
        logger.info(f"Number of layers: {num_layers}")
    
        for idx, (in_channels, out_channels) in enumerate(self.encoder_channels_pair):
            is_last = idx == num_layers -1
            self.encoder.append(
                nn.ModuleList([
                    ResnetBlock(
                        in_channel = in_channels + self.context_channels[idx] if not is_last and idx <= (len(self.context_channels) - 1) else in_channels,
                        out_channel = out_channels,
                        time_embedding=True,
                        time_embedding_channels=self.base_channels,
                    ),
                    ResnetBlock(out_channels, out_channels, time_embedding=True, time_embedding_channels=self.base_channels),
                    LayerNorm(out_channels),
                    LinearAttention(out_channels, head_dimention=4),
                    DownSampling(
                        out_channels, out_channels
                    ) if not is_last else nn.Identity(),
                ])
            )

        bottleneck_in_channel = self.layer_channels[-1]
        self.bottleneck.append(
            nn.ModuleList([
                ResnetBlock(
                    in_channel = bottleneck_in_channel,
                    out_channel = bottleneck_in_channel,
                    time_embedding=True,
                    time_embedding_channels=self.base_channels,
                ),
                LayerNorm(bottleneck_in_channel),
                LinearAttention(bottleneck_in_channel, head_dimention=4),
                ResnetBlock(bottleneck_in_channel, bottleneck_in_channel, time_embedding=True, time_embedding_channels=self.base_channels),
            ])
        )

        for idx, (in_channels, out_channels) in enumerate(self.decoder_channels_pair):
            is_last = idx == num_layers -1
            self.decoder.append(
                nn.ModuleList([
                    ResnetBlock(
                        in_channel = in_channels*2,
                        out_channel = out_channels,
                        time_embedding=True,
                        time_embedding_channels=self.base_channels,
                    ),
                    ResnetBlock(out_channels, out_channels, time_embedding=True, time_embedding_channels=self.base_channels),
                    LayerNorm(out_channels),
                    LinearAttention(out_channels, head_dimention=4),
                    UpSampling(
                        out_channels, out_channels
                    ) if not is_last else nn.Identity(),
                ])
            )

        last_layer_channel = self.decoder_channels_pair[-1][1]
        self.decoder. append(
            nn.Sequential(
                LayerNorm(in_dimension=last_layer_channel),
                nn.Conv2d(in_channels=last_layer_channel, out_channels=self.output_channels, kernel_size=5, padding=2),
            )
        )

    def encode(self, input_tensor, time_tensor, context_tensor):
        original_input = []

        for idx, (res_layer1, res_layer2, layer_norm, attention, downsample) in enumerate(self.encoder):
            logger.info(f"Input tensor {idx}: {input_tensor.shape}")
            logger.info(f"time_tensor: {time_tensor.shape}")

            # Xử lý context tensor nếu có
            context_valid = (
                idx < len(self.context_channels) and
                idx < len(context_tensor) and
                check_valid(context_tensor[idx])
            )

            if idx < len(self.context_channels):
                if context_valid:
                    logger.info("There is context tensor")
                else:
                    logger.info(f"idx: {idx} - No valid context tensor")

            if context_valid:
                logger.info(f"input_tensor dim 1: {input_tensor.shape[1]}")
                logger.info(f"context_tensor[{idx}] dim 1: {context_tensor[idx].shape[1]}")
                cat_tensor = torch.cat([input_tensor, context_tensor[idx]], dim=1)
            else:
                cat_tensor = input_tensor

            logger.info(f"input_tensor: {input_tensor.shape}")
            logger.info(f"cat_tensor: {cat_tensor.shape}")

            # Forward qua các lớp
            input_tensor = res_layer1(cat_tensor, time_tensor)
            logger.info(f"After res_layer1: {input_tensor.shape}")

            input_tensor = res_layer2(input_tensor, time_tensor)
            input_tensor = layer_norm(input_tensor)
            input_tensor = attention(input_tensor)

            # Lưu lại kết quả
            original_input.append(input_tensor)

            # Downsample
            input_tensor = downsample(input_tensor)
            logger.info(f"Encoder Layer {idx}: {input_tensor.shape}")

            # if idx == len(self.encoder_channels_pair) - 1:
            #     original_input.append(input_tensor)

        return input_tensor, original_input


    def bottle_neck(self, input_tensor, time_tensor):
        for res_layer1, layer_norm, attention, res_layer2 in self.bottleneck:
            input_tensor = res_layer1(input_tensor, time_tensor)
            input_tensor = layer_norm(input_tensor)
            input_tensor = attention(input_tensor)
            input_tensor = res_layer2(input_tensor, time_tensor)
            logger.info(f"Bottleneck Layer: {input_tensor.shape}")

        return input_tensor

    def decode(self, input_tensor, time_tensor, original_input):
        for idx, (res_layer1, res_layer2, layer_norm, attention, upsample) in enumerate(self.decoder[:-1]):
            logger.info(f"Decoder Layer {idx}: {input_tensor.shape}")
            logger.info(f"time_tensor: {time_tensor.shape}")

            cat_tensor = input_tensor
            if original_input and original_input[-1].shape[2] == input_tensor.shape[2]:
                skip_tensor = original_input.pop(-1)
                logger.info(f"original_input used for skip connection: {skip_tensor.shape}")
                cat_tensor = torch.cat([input_tensor, skip_tensor], dim=1)

            logger.info(f"input_tensor: {input_tensor.shape}")
            logger.info(f"cat_tensor: {cat_tensor.shape}")

            # Apply decoder layers
            input_tensor = res_layer1(cat_tensor, time_tensor)
            logger.info(f"After res_layer1: {input_tensor.shape}")

            input_tensor = res_layer2(input_tensor, time_tensor)
            input_tensor = layer_norm(input_tensor)
            input_tensor = attention(input_tensor)
            input_tensor = upsample(input_tensor)

            # Resize if necessary to match next skip connection
            if idx < len(self.decoder_channels_pair) - 1 and original_input:
                expected_shape = original_input[-1].shape[2:]
                current_shape = input_tensor.shape[2:]
                if current_shape != expected_shape:
                    input_tensor = F.interpolate(
                        input_tensor,
                        size=expected_shape,
                        mode='bilinear',
                        align_corners=False
                    )

            logger.info(f"Decoder Layer {idx}: {input_tensor.shape}")

        return self.decoder[-1](input_tensor)
    
    def forward(self, input_tensor, time_tensor, context_tensor=None):
        if self.time_embedding:
            time_tensor = self.time_embedding_layer(time_tensor.unsqueeze(1)).squeeze(1)

        if context_tensor is None:
            context_tensor = [torch.zeros(1, 1, 1, 1)] * len(self.encoder_channels_pair)
 
        encode_input, original_input = self.encode(input_tensor, time_tensor, context_tensor)

        for thing in original_input:
            logger.info(f"Original input shape: {thing.shape}")

        bottleneck_output = self.bottle_neck(encode_input, time_tensor)
        output = self.decode(bottleneck_output, time_tensor, original_input)

        return output
            

# # Test the UnetModule class
# if __name__ == "__main__":
#     # Create a random input tensor with shape (batch_size, channels, height, width)
#     input_tensor = torch.randn(1, 3, 1200, 600)  # Example input tensor

#     # Create an instance of the UnetModule class
#     unet_module = UnetModule()

#     logger.critical("UnetModule initialized successfully.")
    
#     logger.info("Layers in the UnetModule:")
#     for layer in unet_module.children():
#         logger.info(layer)

#     context_tensor = [
#         torch.randn(1, 3, 1200, 600),  # Example context tensor for the first layer
#         torch.randn(1, 64, 600, 300),   # Example context tensor for the second layer
#         torch.randn(1, 128, 300, 150),  # Example context tensor for the third layer
#         torch.randn(1, 192, 150, 75),   # Example context tensor for the fourth layer
#     ]

#     output = unet_module(input_tensor, time_tensor=torch.randn(1, 1), context_tensor=context_tensor)
#     logger.critical(f"Unet module worked successfully. Output shape: {output.shape}")
