import os
import sys
import time

from network_components.utils import extract
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np

from compressor.compressor import Compressor
from decompressor.unet_module import UnetModule

from LOGGER import setup_logger

import torch
import torch.nn as nn
from torch.nn import functional as F

import lpips

from PIL import Image


class DiffusionManager(nn.Module):
    def __init__(
        self, 
        encoder=Compressor, 
        u_net=UnetModule,
        num_timesteps=1000,
        bpp_loss_weight=0.2,
        lpips_loss_weight=0.55,
        mse_loss_weight=1,
        device=None
    ):
        super(DiffusionManager, self).__init__()
        # Set device based on provided parameter or availability
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DiffusionManager initialized on device: {self.device}")
        
        self.train_status = True
        
        # Initialize encoder and U-Net
        self.encoder = encoder
        self.u_net = u_net
        
        # Ensure model components are on the correct device
        if isinstance(self.encoder, nn.Module):
            self.encoder.to(self.device)
        if isinstance(self.u_net, nn.Module):
            self.u_net.to(self.device)
        
        self.num_timesteps = num_timesteps

        # Generate schedules
        self.betas_schedule = self.cosine_beta_schedule(num_timesteps)
        self.alphas_schedule = self.alpha_schedule(self.betas_schedule)
        self.alphas_cumprod = torch.tensor(np.cumprod(self.alphas_schedule, axis=0), 
                                          dtype=torch.float32, device=self.device)

        # Pre-compute values for efficiency
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.inv_ac = 1 / self.alphas_cumprod
        self.sqrt_omac = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_inv_ac = torch.sqrt(self.inv_ac)
        self.sqrt_inv_ac_m1 = torch.sqrt(self.inv_ac - 1)

        # Initialize LPIPS loss and move to device
        self.lpips_loss = lpips.LPIPS(net='alex', verbose=False, eval_mode=False if self.train_status else True)
        self.lpips_loss.to(self.device)

        # Loss weights
        self.bpp_loss_weight = bpp_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.mse_loss_weight = mse_loss_weight
        
        # Move all parameters to specified device
        self.to(self.device)

    def cosine_beta_schedule(self, timesteps, s=0.008):
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
        # Move input to device
        # input_start = input_start.to(self.device)
        # step_t = step_t.to(self.device)
        
        # Create noise by formulas: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        if noise is None:
            noise = torch.randn_like(input_start, device=self.device)

        # Tạo nhiễu theo công thức: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        sqrt_ac = self.sqrt_ac[step_t].view(-1, 1, 1, 1)
        sqrt_omac = self.sqrt_omac[step_t].view(-1, 1, 1, 1)
        x_t = sqrt_ac * input_start + sqrt_omac * noise

        return x_t, noise
    
    def denoise_to_x0(self, input_step_t, step_t, estimate_noise):
        # Move inputs to device
        # input_step_t = input_step_t.to(self.device)
        # step_t = step_t.to(self.device)
        # estimate_noise = estimate_noise.to(self.device)
        
        sqrt_inv_ac = self.sqrt_inv_ac[step_t].view(-1, 1, 1, 1)

        # estimate_x0 = sqrt_inv_ac * (input_step_t) - self.sqrt_omac[step_t].view(-1, 1, 1, 1) * estimate_noise
        estimate_x0 = extract(self.sqrt_inv_ac, step_t, input_step_t.shape)*input_step_t - extract(self.sqrt_inv_ac_m1, step_t, input_step_t.shape)*estimate_noise
        return estimate_x0
    
    def loss_fn(self, noise_input, noise, step_t, estimate_noise):
        # Move inputs to device
        # noise_input = noise_input.to(self.device)
        # step_t = step_t.to(self.device)
        # estimate_noise = estimate_noise.to(self.device)
        
        # Tính toán loss theo công thức: L = ||x_0 - x_0_hat||^2
        # Trong đó x_0_hat = x_t - sqrt(1 - alpha_cumprod) * noise
        estimate_x0 = self.denoise_to_x0(noise_input, step_t, estimate_noise)
        loss = F.mse_loss(noise, estimate_noise)

        return loss, estimate_x0
    
    def forward(self, input_tensor):
        # Move input to device
        # input_tensor = input_tensor.to(self.device)
        
        batch_size = input_tensor.size(0)

        # Generate random bitrate conditioning
        bitrate_scalar_batch = torch.full((batch_size,), torch.rand(1).item(), device=self.device)

        # Run encoder
        encoder_output = self.encoder(input_tensor, bitrate_condition=bitrate_scalar_batch)

        context, bpp, _, _ = encoder_output["output"], encoder_output["bpp"], encoder_output["quantize_latent"], encoder_output["quantize_hyper_latent"]

        # Generate random timesteps
        timesteps_tensor = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        # timesteps_tensor = torch.full((batch_size,), self.num_timesteps - 1, device=self.device).long()

        # Create noisy input
        noise_input_tensor, noise = self.create_noise(input_tensor, timesteps_tensor)

        # # # convert first image in noise_input_tensor to image and show
        # image = noise_input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        # image = (image * 0.5 + 0.5).clip(0, 1)  # Chuyển về [0, 1]
        # image = Image.fromarray((image * 255).astype(np.uint8))
        # image.show()

        # Convert to float for U-Net
        timesteps_tensor_float = timesteps_tensor.float() / self.num_timesteps

        # Run U-Net to predict noise
        predicted_noise = self.u_net(noise_input_tensor, timesteps_tensor_float, context)

        # Convert back to long for loss calculation
        timesteps_tensor = timesteps_tensor.long()

        # Calculate losses
        mse_loss, estimate_x0 = self.loss_fn(noise_input_tensor, noise, timesteps_tensor, predicted_noise)
        # get one estimate x0
        
        # Calculate LPIPS loss between input_tensor and estimate_x0
        lpips_loss = self.lpips_loss(input_tensor, estimate_x0).mean()
        bpp_loss = bpp.mean()

        # Dynamic bitrate weight
        self.bpp_loss_weight = 2 ** (3 * bitrate_scalar_batch) * 5e-4
        self.bpp_loss_weight = self.bpp_loss_weight.mean()

        # Calculate total loss
        total_loss = (
            self.mse_loss_weight * mse_loss +0
            # 0.3 * lpips_loss
            # self.bpp_loss_weight * bpp_loss
        )

        # Calculate prior probability loss
        prior_probability_loss = self.encoder.prior_probability_loss()

        loss_dict = {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
            "bpp_loss": bpp_loss,
            "prior_probability_loss": prior_probability_loss,
        }

        estimate_x0 = estimate_x0[0].unsqueeze(0)

        return total_loss, prior_probability_loss, estimate_x0, loss_dict, noise_input_tensor
    

    def evaluate_ddim(
        self,
        original_image,
        start_noise,
        denoise_steps = 30,
    ):
        batch_size = start_noise.size(0)
        time_steps = torch.linspace(0, self.num_timesteps - 1, denoise_steps).long().to(self.device)
        self.test_ac = self.alphas_cumprod[time_steps]
        self.test_ac_prev = torch.cat([torch.ones(1, device=self.device), self.test_ac[:-1]], dim=0)
        self.test_sqrt_ac = torch.sqrt(self.test_ac)
        self.test_sqrt_ac_prev = torch.sqrt(self.test_ac_prev)
        self.test_inv_ac = 1 / self.test_ac
        self.test_inv_ac_prev = 1 / self.test_ac_prev
        self.test_sqrt_omac = torch.sqrt(1 - self.test_ac)
        self.test_sqrt_omac_prev = torch.sqrt(1 - self.test_ac_prev)
        self.test_sqrt_inv_ac = torch.sqrt(self.test_inv_ac)
        self.test_sqrt_inv_ac_prev = torch.sqrt(self.test_inv_ac_prev)
        self.test_sqrt_inv_ac_m1 = torch.sqrt(self.test_inv_ac - 1)
        self.test_sqrt_inv_ac_m1_prev = torch.sqrt(self.test_inv_ac_prev - 1)

        # self.sigma = torch.sqrt(
        #     (1 - self.test_ac_prev) / (1 - self.test_ac) * (1 - self.test_ac_prev / self.test_ac)
        # )

        encoder_output = self.encoder(original_image)

        context, bpp, _, _ = encoder_output["output"], encoder_output["bpp"], encoder_output["quantize_latent"], encoder_output["quantize_hyper_latent"]

        denoise_result = None
        for step_idx in reversed(range(0, denoise_steps)):
            step_t = torch.full((batch_size,), step_idx, device=self.device).long()
            print(f"Step {step_idx}/{denoise_steps-1}")
            t_normalized = step_t.float() / denoise_steps
            predicted_noise_stept = self.u_net(start_noise, t_normalized, context)

            reconstruction_stept = self.ddim_denoise_step_t(
                start_noise, step_t, predicted_noise_stept
            )

            next_step_noise = reconstruction_stept
            # next_step_noise = self.test_sqrt_ac_prev[step_t] * next_step_noise + self.test_sqrt_omac_prev[step_t] * predicted_noise_stept
            next_step_noise = extract(self.test_sqrt_ac_prev, step_t, start_noise.shape) * next_step_noise \
                            + extract(self.test_sqrt_omac_prev, step_t, start_noise.shape) * predicted_noise_stept
            # Draw first image in start_noise before updating it
            first_image = next_step_noise[0].permute(1, 2, 0).detach().cpu().numpy()
            first_image = (first_image * 0.5 + 0.5).clip(0, 1)
            first_image = Image.fromarray((first_image * 255).astype(np.uint8))
            first_image.show()

            time.sleep(300)

            start_noise = next_step_noise

            image = reconstruction_stept[0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 0.5 + 0.5).clip(0, 1)
            image = (image * 255).astype(np.uint8)

            # Chuyển sang định dạng BGR cho OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Save the first, middle, and last constructed images in a folder with differentiated names
            save_dir = "denoised_results"
            os.makedirs(save_dir, exist_ok=True)
            if step_idx == denoise_steps - 1:
                filename = f"first_denoised_image_{int(time.time())}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, image_bgr)
            if step_idx == denoise_steps // 2:
                filename = f"middle_denoised_image_{int(time.time())}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, image_bgr)
            if step_idx == 0:
                filename = f"last_denoised_image_{int(time.time())}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, image_bgr)
                denoise_result = reconstruction_stept

            del image_bgr
            del image

        return denoise_result
        
       
    def ddim_denoise_step_t(self, input_step_t, step_t, estimate_noise):
        # Get 1/√(α_t)
        sqrt_inv_alpha = extract(self.test_sqrt_inv_ac, step_t, input_step_t.shape)
        # Get √(1/α_t - 1) - this is mathematically equivalent to √(1-α_t)/√(α_t)
        sqrt_one_minus_alpha_over_alpha = extract(self.test_sqrt_inv_ac_m1, step_t, input_step_t.shape)
        
        # Correctly predict x₀ using your formula
        predicted_x0 = sqrt_inv_alpha * input_step_t - sqrt_one_minus_alpha_over_alpha * estimate_noise
        
        return predicted_x0
    
    def draw_image(input):
        image = input[0].cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.show()
        time.sleep(1)
        image.close()

