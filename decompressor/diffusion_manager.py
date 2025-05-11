import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from compressor.compressor import Compressor
from decompressor.unet_module import UnetModule

from LOGGER import setup_logger

import torch
import torch.nn as nn
from torch.nn import functional as F

import lpips

class DiffusionManager(nn.Module):
    def __init__(
        self, 
        encoder=Compressor, 
        u_net=UnetModule,
        num_timesteps=1000,
        bpp_loss_weight=0.2,
        lpips_loss_weight=0.2,
        mse_loss_weight=0.6,
    ):
        super(DiffusionManager, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train = True
        self.encoder = encoder
        self.u_net = u_net
        self.num_timesteps = num_timesteps

        self.betas_schedule = self.cosine_beta_schedule(num_timesteps)
        self.alphas_schedule = self.alpha_schedule(self.betas_schedule)
        self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas_schedule, axis=0), dtype=torch.float32)

        # ac = alpha_cumprod
        # inv_ac = 1/ac
        # omac = 1 minus ac
        # m1 = minus 1
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.inv_ac = 1 / self.alphas_cumprod
        self.sqrt_omac = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_inv_ac = torch.sqrt(self.inv_ac)
        self.sqrt_inv_ac_m1  = torch.sqrt(self.inv_ac - 1)


        self.lpips_loss = lpips.LPIPS(net='alex', verbose=False, eval_mode=False if self.train == True else True)

        self.bpp_loss_weight = bpp_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.mse_loss_weight = mse_loss_weight
        # self.register_buffer("betas", torch.tensor(self.betas_schedule, dtype=torch.float32))
        # self.register_buffer("alphas", torch.tensor(self.alphas_schedule, dtype=torch.float32))
        # self.register_buffer("alphas_cumprod", torch.tensor(self.alphas_cumpod, dtype=torch.float32))


    def cosine_beta_schedule(self, timesteps, s=0.08):
        """
        Tạo lịch trình beta dựa trên hàm cosine theo phương pháp đề xuất 
        trong paper "Improved Denoising Diffusion Probabilistic Models":
        https://openreview.net/forum?id=-NEXDKk8gZ

        Args:
            timesteps (int): Số bước thời gian của mô hình diffusion.
            s (float): Tham số điều chỉnh để tránh alpha bắt đầu từ 0 (mặc định 0.008).

        Returns:
            np.ndarray: Mảng các giá trị beta có độ dài `timesteps`.
        """
        steps = timesteps + 1  # Thêm 1 để tính alpha_cumprod tại mỗi điểm phân cách
        x = np.linspace(0, steps, steps)  # Các mốc thời gian từ 0 đến timesteps

        # Tính giá trị alpha_cumprod theo công thức cosine
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2

        # Chuẩn hóa để alphas_cumprod[0] = 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Tính beta_t = 1 - (alpha_t / alpha_{t-1})
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return np.clip(betas, a_min=0.0001, a_max=0.999)
    
    def alpha_schedule(self, beta_schedule):
        alpha_schedule = 1.0 - beta_schedule
        return alpha_schedule
    
    def create_noise(self, input_start, step_t, noise=None):
        # Create nosie by formulas: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        if noise is None:
            noise = torch.randn_like(input_start)

        # Tạo nhiễu theo công thức: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        sqrt_ac = self.sqrt_ac[step_t].view(-1, 1, 1, 1)
        sqrt_omac = self.sqrt_omac[step_t].view(-1, 1, 1, 1)
        x_t = sqrt_ac * input_start + sqrt_omac * noise

        return x_t
    
    def denoise_to_x0(self, input_step_t, step_t, estimate_noise):
        sqrt_inv_ac = self.sqrt_inv_ac[step_t].view(-1, 1, 1, 1)

        estimate_x0 = sqrt_inv_ac * (input_step_t - self.sqrt_omac[step_t].view(-1, 1, 1, 1) * estimate_noise)
        return estimate_x0
    
    def loss_fn(self, input_step_t, step_t, estimate_noise):
        # Tính toán loss theo công thức: L = ||x_0 - x_0_hat||^2
        # Trong đó x_0_hat = x_t - sqrt(1 - alpha_cumprod) * noise
        # loss = ||x_0 - x_0_hat||^2
        estimate_x0 = self.denoise_to_x0(input_step_t, step_t, estimate_noise)
        loss = F.mse_loss(estimate_x0, input_step_t)

        # # Draw estimate x0
        # estimate_x0 = estimate_x0.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # estimate_x0 = (estimate_x0 * 0.5 + 0.5).clip(0, 1)  # Chuyển về [0, 1]
        # plt.imshow(estimate_x0)
        # plt.title("Estimate x0")
        # plt.axis("off")
        # plt.show()
        return loss
    
    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)

        bitrate_scalar_batch = torch.rand(batch_size, device=input_tensor.device)

        encoder_output = self.encoder(input_tensor, bitrate_condition=bitrate_scalar_batch)

        context, bpp, _, _ = encoder_output["output"], encoder_output["bpp"], encoder_output["quantize_latent"], encoder_output["quantize_hyper_latent"]

        # Tạo nhiễu ngẫu nhiên
        timesteps_tensor = torch.randint(0, self.num_timesteps, (batch_size,), device=input_tensor.device).long()

        noise_input_tensor = self.create_noise(input_tensor, timesteps_tensor)

        timesteps_tensor = timesteps_tensor.float()

        predicted_noise = self.u_net(noise_input_tensor, timesteps_tensor, context)

        timesteps_tensor = timesteps_tensor.long()

        mse_loss = self.loss_fn(noise_input_tensor, timesteps_tensor, predicted_noise)
        lpips_loss = self.lpips_loss(input_tensor, predicted_noise).mean()
        bpp_loss = bpp.mean()

        self.bpp_loss_weight = 2 ** (3 * bitrate_scalar_batch) * 5e-4

        # Tính toán tổng hợp các loss
        total_loss = (
            self.mse_loss_weight * mse_loss +
            self.lpips_loss_weight * lpips_loss +
            self.bpp_loss_weight * bpp_loss
        )

        prior_probability_loss = self.encoder.prior_probability_loss()

        return total_loss, prior_probability_loss
    

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Khởi tạo diffusion manager
compressor = Compressor(
    in_channel=3,
    out_channel=3,
    base_channel=64,
    bitrate_conditional=True,
)

unet_module = UnetModule()

diffusion = DiffusionManager(
    encoder=compressor,
    u_net=unet_module,
)

# Load ảnh
img_path = "dogpicture.png"
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize để xử lý nhanh
    transforms.ToTensor(),          # Chuyển sang tensor
])
img_tensor = transform(image).unsqueeze(0)  # Thêm batch dimension (1, 3, H, W)

output = diffusion(img_tensor)

print("Output 0:", output[0])  # In ra đầu ra
print("Output 1:", output[1])  # In ra đầu ra


# # Tạo nhiễu ngẫu nhiên và hiển thị ảnh với nhiễu
# noise = diffusion.create_noise(img_tensor, step_t=50, noise=None)
# print("Noise shape:", noise.shape)
# noised_image = noise
# noised_image = noised_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
# noised_image = (noised_image * 0.5 + 0.5).clip(0, 1)  # Chuyển về [0, 1]

# plt.subplot(1, 2, 1)
# plt.imshow(noised_image)
# plt.title("Image with Noise")
# plt.axis("off")

# plt.tight_layout()
# # plt.show()



# # Tạo nhiễu ngẫu nhiên và hiển thị ảnh với nhiễu
# noise = diffusion.create_noise(img_tensor, step_t=50, noise=None)
# print("Noise shape:", noise.shape)
# noised_image = noise
# noised_image = diffusion.denoise_to_x0(noised_image, step_t=500, estimate_noise=noise)
# noised_image = noised_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
# noised_image = (noised_image * 0.5 + 0.5).clip(0, 1)  # Chuyển về [0, 1]

# plt.subplot(1, 2, 2)
# plt.imshow(noised_image)
# plt.title("Image with Noise")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# # Tạo nhiễu tại các bước thời gian
# timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]

# plt.figure(figsize=(18, 6))
# for idx, t in enumerate(timesteps_to_show):
#     t_tensor = torch.tensor([t], dtype=torch.long)
#     noised = diffusion.create_noise(img_tensor, t_tensor)

#     # Đưa tensor về ảnh để hiển thị
#     img_np = noised.squeeze(0).detach().cpu()
#     img_np = (img_np * 0.5 + 0.5).clamp(0, 1)  # Chuyển lại về [0, 1]
#     img_np = img_np.permute(1, 2, 0).numpy()

#     plt.subplot(1, len(timesteps_to_show), idx + 1)
#     plt.imshow(img_np)
#     plt.title(f"Step {t}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()
