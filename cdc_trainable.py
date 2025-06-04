import os
import sys
import time
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import datetime

from network_components.layer_normalization import LayerNorm
from network_components.linear_attention import LinearAttention
from network_components.resize_input import DownSampling, UpSampling

# Modern autocast import
from torch.amp import autocast

from network_components.utils import extract
from PIL import Image
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, large=False, time_embedding=False, time_embedding_channels=None):
        super(ResnetBlock, self).__init__()
        self.kernel_size = 5 if large else 3

        self.mlp = None
        if time_embedding is not False:
            self.mlp = nn.Sequential(
                nn.Linear(time_embedding_channels, out_channel),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(out_channel, out_channel),
            )

        # Make this a proper ModuleList
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            LayerNorm(out_channel),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LayerNorm(out_channel),
            nn.ReLU()
        )

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, input, time_tensor=None):
        ori = input
        if time_tensor is not None:
            time_tensor = self.mlp(time_tensor)

        # Update the forward logic to use the new module organization
        input = self.block1(input)
        if time_tensor is not None:
            input = input + time_tensor.view(time_tensor.shape[0], time_tensor.shape[1], 1, 1)
        input = self.block2(input)
        shortcut = self.shortcut(ori)
        return input + shortcut

class CDCTrainable(nn.Module):
    def __init__(self, device):
        super(CDCTrainable, self).__init__()
        self.extractor_shape = None
        self.hyper_downscale_shape = None
        self.device = device

        self.create_variable_for_extractor_phase()

        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        context = self.extractor_forward(dummy_input)
        context_channels = [c.shape[1] for c in context]

        self.create_variable_image_generator_phase(
            context_channels=context_channels,
        )

        self.create_noise_phase_variable(timesteps=1000)
        pass 

    def create_variable_for_extractor_phase(
        self,
        in_channels=3,
        channel_multiplier=[1, 2, 3, 4, 5, 6],
        base_channels=32,
        hyper_multiplier=[1, 1, 1],
    ):
        self.extractor = nn.ModuleList()
        self.hyper_encoder = nn.ModuleList()
        self.hyper_decoder = nn.ModuleList()
        self.feature_generator = nn.ModuleList()
        self.image_generator_unet = nn.ModuleList()
        # ========== Encoder & feature_generator Variable ==========#
        extractor_encode_channels = [in_channels]
        for multiplier in channel_multiplier:
            extractor_encode_channels.append(base_channels * multiplier)

        self.eec_pairs = [] # Extractor Encoder Channels Pair
        for i in range(len(extractor_encode_channels) - 1):
            self.eec_pairs.append(
                (extractor_encode_channels[i], extractor_encode_channels[i + 1])
            )

        feature_gen_channels = extractor_encode_channels[::-1]
        self.fgc_pairs = [] # Feature Generator Channels Pair
        for i in range(len(feature_gen_channels) - 1):
            self.fgc_pairs.append(
                (feature_gen_channels[i], feature_gen_channels[i + 1])
            )

        #========= Hyper Encoder & Decoder Variable ==========#
        hyper_encode_channels = [extractor_encode_channels[-1]]
        for i in range(len(hyper_multiplier)):
            hyper_encode_channels.append(
                extractor_encode_channels[-1] * hyper_multiplier[i]
            )
               
        self.hec_pairs = [] # Hyper Encoder Channels Pair
        for i in range(len(hyper_multiplier)):
            self.hec_pairs.append(
                (hyper_encode_channels[i], hyper_encode_channels[i] * hyper_multiplier[i])
            )

        backward_hyperdecode_channels = hyper_encode_channels[::-1]
        self.hdc_pairs = [] # Hyper Decoder Channels Pair
        for i in range(len(backward_hyperdecode_channels) - 1):
            if i == len(backward_hyperdecode_channels) - 2:
                self.hdc_pairs.append(
                    (backward_hyperdecode_channels[i], backward_hyperdecode_channels[i + 1] * 2)
                )
            else:
                self.hdc_pairs.append(
                    (backward_hyperdecode_channels[i], backward_hyperdecode_channels[i + 1])
                )


        del extractor_encode_channels
        del feature_gen_channels
        del hyper_encode_channels
        del backward_hyperdecode_channels
        gc.collect()
        torch.cuda.empty_cache()

        self.create_network_feature_phase()
        self.extractor = self.extractor.to(self.device)
        self.hyper_encoder = self.hyper_encoder.to(self.device)
        self.hyper_decoder = self.hyper_decoder.to(self.device)
        self.feature_generator = self.feature_generator.to(self.device)

    def create_network_feature_phase(self):
        for idx, (dim_in, dim_out) in enumerate(self.eec_pairs):
            block = nn.ModuleList([
                ResnetBlock(dim_in, dim_out, large=True if idx==0 else False),
                DownSampling(dim_out, dim_out)
            ])
            self.extractor.append(block)

        num_layers = len(self.fgc_pairs)
        for idx, (dim_in, dim_out) in enumerate(self.fgc_pairs):
            is_last = True if idx == num_layers - 1 else False
            cond_dim = dim_in if is_last else dim_out
            block = nn.ModuleList([
                ResnetBlock(dim_in, cond_dim),
                UpSampling(cond_dim, dim_out),
            ])
            self.feature_generator.append(block)

        num_layers = len(self.hec_pairs)
        for idx, (dim_in, dim_out) in enumerate(self.hec_pairs):
            is_last = True if idx == num_layers - 1 else False
            block = nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, kernel_size=3 if idx==0 else 5, padding=2 if idx==0 else 2, stride=1 if idx==0 else 2),
                nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
            ])
            self.hyper_encoder.append(block)

        num_layers = len(self.hdc_pairs)
        for idx, (dim_in, dim_out) in enumerate(self.hdc_pairs):
            is_last = True if idx == num_layers - 1 else False
            block = nn.ModuleList([
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=5, stride=2, padding=2, output_padding=1) if not is_last \
                else nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
            ])
            self.hyper_decoder.append(block)

    # -------------------- Feature Extractor Phase -------------------- #

    def feature_extract(self, image_tensor):
        if self.extractor_shape is None:
            self.extractor_shape = [image_tensor.shape]

        current_features = image_tensor
        for idx, block in enumerate(self.extractor):
            self.extractor_shape.append(current_features.shape)
            for layer in block:
                current_features = layer(current_features)

        latent_variable = current_features

        return latent_variable

    def feature_generate(self, latent_variable):
        features_output = []

        current_features = latent_variable
        for idx, block in enumerate(self.feature_generator):
            for layer in block:
                current_features = layer(current_features)
            if (current_features.shape != self.extractor_shape[-(idx + 1)]):
                current_features = F.interpolate(current_features, size=self.extractor_shape[-(idx + 1)][2:], mode='bilinear', align_corners=False)

            features_output.append(current_features)
        return features_output[::-1]
    
    # ---------------- Image Generator Phase -------------------- #

    def create_variable_image_generator_phase(
        self,
        context_channels=[3, 16, 32, 64, 128, 256],
        base_channels=32,
        channel_multiplier=[1, 2, 3, 4, 5 ,6, 7, 8],
        time_embedding_channels=128,
    ):
        self.unet_encoder = nn.ModuleList() 
        self.unet_bottleneck = nn.ModuleList()
        self.unet_decoder = nn.ModuleList()
        self.num_layers = len(channel_multiplier)
        self.context_channels = context_channels
        self.base_channels = context_channels
        self.tec = time_embedding_channels # Time Embedding Channels

        dims = [3, *(map(lambda m: base_channels * m, channel_multiplier))]
        in_out = list(zip(dims[:-1], dims[1:]))

        print("UNet Encoder Input-Output Pairs:", in_out)

        self.uec_pairs = []  # UNet Encoder Channels Pair     
        for idx, (layer_in, layer_out) in enumerate(in_out):
            is_last = True if idx == len(channel_multiplier) - 1 else False
            self.uec_pairs.append(
                (context_channels[idx] + layer_in if not is_last and idx < len(context_channels) else layer_in, layer_out)
            )

        print("UNet Encoder Channels Pairs:", self.uec_pairs)

        self.udc_pairs = []  # UNet Decoder Channels Pair
        reversed_in_out = list(reversed(in_out[:]))
        for idx, (layer_in, layer_out) in enumerate(reversed_in_out):
            is_last = idx == len(reversed_in_out) - 1
            out_channels = 3 if is_last else layer_in
            self.udc_pairs.append(
                (layer_out * 2, out_channels)
            )

        print("UNet Decoder Channels Pairs:", self.udc_pairs)

        print("Context Channels:", self.context_channels)

        self.create_network_image_generator_phase()

        self.unet_encoder = self.unet_encoder.to(self.device)
        self.unet_bottleneck = self.unet_bottleneck.to(self.device)
        self.unet_decoder = self.unet_decoder.to(self.device)
        self.time_embedding = self.time_embedding.to(self.device) 


    def create_network_image_generator_phase(self):
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )

        # self.time_embedding = SinusoidalPosEmb(128)

        for idx, (layer_in, layer_out) in enumerate(self.uec_pairs):
            is_last = True if idx == self.num_layers - 1 else False
            block = nn.ModuleList([
                ResnetBlock(layer_in, layer_in, time_embedding=True, time_embedding_channels=self.tec),
                ResnetBlock(layer_in, layer_out, time_embedding=True, time_embedding_channels=self.tec),
                LayerNorm(layer_out),
                LinearAttention(layer_out),
                DownSampling(layer_out, layer_out) if not is_last else nn.Identity(),
            ])
            # setattr(self, f'unet_up_layer{idx}', block)
            self.unet_encoder.append(block)

        bottleneck_in_channel = self.uec_pairs[-1][1]
        self.unet_bottleneck.append(
            nn.ModuleList([
                ResnetBlock(bottleneck_in_channel, bottleneck_in_channel, time_embedding=True, time_embedding_channels=self.tec),
                LayerNorm(bottleneck_in_channel),
                LinearAttention(bottleneck_in_channel),
                ResnetBlock(bottleneck_in_channel, bottleneck_in_channel, time_embedding=True, time_embedding_channels=self.tec),
            ])
        )

        for idx, (layer_in, layer_out) in enumerate(self.udc_pairs):
            is_last = True if idx == self.num_layers - 1 else False
            block = nn.ModuleList([
                ResnetBlock(layer_in, layer_in, time_embedding=True, time_embedding_channels=self.tec),
                ResnetBlock(layer_in, layer_out, time_embedding=True, time_embedding_channels=self.tec),
                LayerNorm(layer_out),
                LinearAttention(layer_out),
                UpSampling(layer_out, layer_out) if not is_last else nn.Identity(),
            ])
            # setattr(self, f'unet_down_layer{idx}', block)
            self.unet_decoder.append(block)

        last_layer_channel = self.udc_pairs[-1][1]
        self.unet_decoder.append(
            nn.Sequential(
                LayerNorm(last_layer_channel),
                nn.Conv2d(last_layer_channel, 3, kernel_size=7, stride=1, padding=3),
                # nn.ReLU(),
                # nn.Conv2d(12, 3, kernel_size=1, stride=1, padding=0)
            )
        )

    def unet_forward(self, input_tensor, context_tensor, time_tensor):
        start_time = time.time()
        time_tensor = self.time_embedding(time_tensor)
        cat_layer = [input_tensor]

        for idx, (res1, res2, layer_norm, attention, downsample) in enumerate(self.unet_encoder):
            is_last = True if idx == self.num_layers - 1 else False
            if idx < len(context_tensor):
                cat_tensor = torch.cat([input_tensor, context_tensor[idx]], dim=1)
            else:
                cat_tensor = input_tensor
            input_tensor = res1(cat_tensor, time_tensor)
            input_tensor = res2(input_tensor, time_tensor)
            input_tensor = layer_norm(input_tensor)
            input_tensor = attention(input_tensor)
            cat_layer.append(input_tensor)
            input_tensor = downsample(input_tensor)
        del cat_tensor
        torch.cuda.empty_cache()

        for res1, layer_norm, attention, res2 in self.unet_bottleneck:
            input_tensor = res1(input_tensor, time_tensor)
            input_tensor = layer_norm(input_tensor)
            input_tensor = attention(input_tensor)
            input_tensor = res2(input_tensor, time_tensor)

        for idx, (res1, res2, layer_norm, attention, upsample) in enumerate(self.unet_decoder[:-1]):
            cat_tensor = torch.cat([input_tensor, cat_layer.pop()], dim=1)
            input_tensor = res1(cat_tensor, time_tensor)
            input_tensor = res2(input_tensor, time_tensor)
            input_tensor = layer_norm(input_tensor)
            input_tensor = attention(input_tensor)
            input_tensor = upsample(input_tensor)

            if cat_layer[-1].shape[2:] != input_tensor.shape[2:]:
                input_tensor = F.interpolate(input_tensor, cat_layer[-1].shape[2:], mode='bilinear', align_corners=False)
        del cat_layer
        torch.cuda.empty_cache()
        last_layer = self.unet_decoder[-1]
        input_tensor = last_layer(input_tensor)

        return input_tensor

    def extractor_forward(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        latent_variable = self.feature_extract(image_tensor)
        features_output = self.feature_generate(latent_variable)
        return features_output
    
    # ---------------- Noise part -------------------- #
    def cosine_beta_schedule(self, timesteps=1000, s=0.008):
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
    
    def alpha_schedule(self, beta_schedule):
        alphas = 1.0 - beta_schedule
        return alphas
    
    def create_noise_phase_variable(self, timesteps=1000):
        beta_schedule = self.cosine_beta_schedule(timesteps)
        self.alphas_schedule = self.alpha_schedule(beta_schedule)
        self.ac = torch.tensor(np.cumprod(self.alphas_schedule, axis=0, dtype=np.float32)).to(self.device)
        self.sqrt_ac = torch.sqrt(self.ac)
        self.inv_ac = 1/self.ac 
        self.sqrt_omac = torch.sqrt(1 - self.ac)
        self.sqrt_inv_ac = torch.sqrt(self.inv_ac)
        self.sqrt_inv_ac_m1 = torch.sqrt(self.inv_ac - 1)

    def create_noise_step_t(self, input_tensor, step_t, noise=None):
        # Formula: x_t = sqrt_ac * x_0 + sqrt_omac * noise
        if noise is None:
            noise = torch.randn_like(input_tensor, device=self.device)
        sqrt_ac = extract(self.sqrt_ac, step_t, input_tensor.shape)
        sqrt_omac = extract(self.sqrt_omac, step_t, input_tensor.shape)
        return sqrt_ac * input_tensor + sqrt_omac * noise
    
    def calculate_loss(self, noise_input, noise, step_t, estimated_noise):
        estimated_x0 = self.denoise_step_t(noise_input, step_t, estimated_noise)
        loss = F.mse_loss(noise, estimated_noise)
        return loss, estimated_x0
    
    def denoise_step_t(self, input_tensor, step_t, estimated_noise):
        # Formula: x_0 = sqrt_inv_ac * x_t - sqrt_inv_ac_m1 * estimated_noise
        sqrt_inv_ac = extract(self.sqrt_inv_ac, step_t, input_tensor.shape)
        sqrt_inv_ac_m1 = extract(self.sqrt_inv_ac_m1, step_t, input_tensor.shape)
        return sqrt_inv_ac * input_tensor - sqrt_inv_ac_m1 * estimated_noise

    def forward(self, image_tensor):
        # start_time = time.time()
        context = self.extractor_forward(image_tensor)
        noise = torch.randn_like(image_tensor, device=self.device)
        timesteps_tensor = torch.randint(0, 1000, (image_tensor.shape[0],), device=self.device).long()
        noise_tensor = self.create_noise_step_t(image_tensor, timesteps_tensor, noise=noise)
        float_timesteps_tensor = timesteps_tensor.float().unsqueeze(-1)/1000 # Convert to shape (batch_size, 1)
        predict_noise = self.unet_forward(noise_tensor, context, float_timesteps_tensor)
        mse_loss, estimated_x0 = self.calculate_loss(noise_tensor, noise, timesteps_tensor, predict_noise)
        # print("Forward pass completed in {:.5f} seconds".format(time.time() - start_time))
        estimated_x0 = estimated_x0[0].unsqueeze(0)  # Removeindices = torch.linspace(0, denoise_steps-1) batch dimension for output
        return mse_loss, estimated_x0, noise_tensor
    
    # ================ DDIM Phase ================ #
    def create_ddim_phase_variable(self, denoise_steps = 50, eta=0):
        indices = torch.linspace(0, 999, denoise_steps, device=self.device).long()
        self.ddim_ac = self.ac[indices]
        # np.set_printoptions(precision=5, suppress=True, linewidth=120)
        # print("ddim_ac:", self.ddim_ac.cpu().numpy())
        self.ddim_ac_prev = F.pad(self.ddim_ac[:-1], (1, 0), value=1.0)
        # print("ddim_ac_prev:", self.ddim_ac_prev.cpu().numpy())
        self.ddim_sqrt_ac = torch.sqrt(self.ddim_ac)
        # print("ddim_sqrt_ac:", self.ddim_sqrt_ac.cpu().numpy())
        self.ddim_sqrt_ac_prev = torch.sqrt(self.ddim_ac_prev)
        # print("ddim_sqrt_ac_prev:", self.ddim_sqrt_ac_prev.cpu().numpy())
        self.ddim_omac = 1 - self.ddim_ac
        # print("ddim_omac:", self.ddim_omac.cpu().numpy())
        self.ddim_omac_prev = 1 - self.ddim_ac_prev
        # print("ddim_omac_prev:", self.ddim_omac_prev.cpu().numpy())
        self.ddim_sqrt_omac = torch.sqrt(1 - self.ddim_ac)
        # print("ddim_sqrt_omac:", self.ddim_sqrt_omac.cpu().numpy())
        self.ddim_sqrt_omac_prev = torch.sqrt(1 - self.ddim_ac_prev)
        # print("ddim_sqrt_omac_prev:", self.ddim_sqrt_omac_prev.cpu().numpy())
        self.ddim_sqrt_inv_ac = torch.sqrt(1 / self.ddim_ac)
        # print("ddim_sqrt_inv_ac:", self.ddim_sqrt_inv_ac.cpu().numpy())
        self.ddim_sqrt_inv_ac_prev = torch.sqrt(1 / self.ddim_ac_prev)
        # print("ddim_sqrt_inv_ac_prev:", self.ddim_sqrt_inv_ac_prev.cpu().numpy())
        self.ddim_sqrt_inv_ac_m1 = torch.sqrt(1 / self.ddim_ac - 1)
        # print("ddim_sqrt_inv_ac_m1:", self.ddim_sqrt_inv_ac_m1.cpu().numpy())

    def ddim_denoise_step_t(self, input_tensor, step_t, estimated_noise):
        # Formula: x_0 = sqrt_inv_ac * x_t - sqrt_inv_ac_m1 * estimated_noise
        sqrt_inv_ac = extract(self.ddim_sqrt_inv_ac, step_t, input_tensor.shape)
        sqrt_inv_ac_m1 = extract(self.ddim_sqrt_inv_ac_m1, step_t, input_tensor.shape)
        return sqrt_inv_ac * input_tensor - sqrt_inv_ac_m1 * estimated_noise
    
    def ddim_input_for_nextstep(self, predict_start, predict_noise, step_t):
        a = extract(self.ddim_sqrt_ac_prev, step_t, predict_start.shape) * predict_start
        b = extract(self.ddim_sqrt_omac_prev, step_t, predict_start.shape) * predict_noise 
        return a + b 
    
    @autocast(device_type='cuda', enabled=torch.cuda.is_available())
    @torch.no_grad()
    def ddim_forward(self, image_tensor, denoise_steps=50):
        self.create_ddim_phase_variable(denoise_steps=denoise_steps)
        context = self.extractor_forward(image_tensor)
        start_noise = torch.randn_like(image_tensor, device=self.device)
        
        for step_t in reversed(range(0, denoise_steps)):
            # self.save_sample(start_noise, step=step_t)
            print(f"DDIM Step: {step_t+1}/{denoise_steps}")
            float_timesteps_tensor = torch.full((image_tensor.shape[0],), step_t / (denoise_steps - 1), device=self.device).unsqueeze(-1)  # Convert to shape (batch_size, 1)
            timesteps_tensor = torch.full((image_tensor.shape[0],), step_t, device=self.device).long()  # Convert to shape (batch_size,)
            predict_noise = self.unet_forward(start_noise, context, float_timesteps_tensor)
            predict_x0 = self.ddim_denoise_step_t(start_noise, timesteps_tensor, predict_noise)
            predict_x0 = predict_x0.clamp(0, 1.0)  # Clamp to valid range
            start_noise = self.ddim_input_for_nextstep(predict_x0, predict_noise, timesteps_tensor)
            self.save_sample(predict_x0, step=step_t)
            

    def save_sample(self, tensor, path='./sample/', step=None):
        if not os.path.exists(path):
            os.makedirs(path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if step is not None:
            filename = f'sample_step{step}_{timestamp}.png'
        else:
            filename = f'sample_{timestamp}.png'

        # Ensure tensor is detached and moved to CPU
        tensor = tensor.detach().cpu()

        # If it's a single image (C, H, W), add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Clamp values just in case
        tensor = tensor.clamp(0, 1)

        # Save image
        save_image(tensor, os.path.join(path, filename))

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CDCTrainable(device=device)
model = model.to(device)


# model.create_ddim_phase_variable()

# # # Load input image from file and preprocess
# pil_image = Image.open("image.png").convert("RGB")
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),  # Converts to [0,1]
#     transforms.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1,1]
# ])
# input_tensor = transform(pil_image).unsqueeze(0).to(device)
# # model.eval()
# # model.ddim_forward(input_tensor, denoise_steps=50)

# # # Modern mixed precision
# input_tensor = torch.randn(1, 3, 256, 256, device=device)  # Place directly on device

# # Use modern autocast syntax
# # Simple training phase
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# model.train()
# for epoch in range(1000):  # For demonstration, just 1 epoch
#     start_time = time.time()
#     optimizer.zero_grad()
#     with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
#         loss, estimate = model(input_tensor)
#     print(f"Loss: {loss.item()}")
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1} completed in {time.time() - start_time:.5f} seconds")

# For model summary with torchsummary
try:
    from torchsummary import summary
    # Generate model summary for the CDCTrainable model
    summary(model, input_size=(3, 256, 256), device=str(device))
except ImportError:
    print("Install torchsummary for model summary: pip install torchsummary")

# # Calculate actual memory consumption
# if torch.cuda.is_available():
#     mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
#     mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
#     print(f"CUDA memory allocated: {mem_allocated:.2f} MB")
#     print(f"CUDA memory reserved: {mem_reserved:.2f} MB")
# else:
#     import psutil, os
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info().rss / (1024 ** 2)
#     print(f"CPU memory usage: {mem_info:.2f} MB")
