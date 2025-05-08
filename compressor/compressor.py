import logging
import sys
import os

# Add parent directory to path so Python can find the network_components module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LOGGER import setup_logger
from network_components.utils import NormalDistribution, quantize

import torch
import torch.nn as nn

from network_components.hyper_prior import HyperPrior
from network_components.resize_input import DownSampling, UpSampling
from network_components.resnet_block import ResnetBlock
from network_components.variable_bitrate_condition import VariableBitrateCondition

logger = setup_logger(name='CompressorLogger', log_file='compressor.log', level=logging.WARNING)

class Compressor(nn.Module):
    """
    Define bas comressor class taking the action of compressing and decompressing the input to/from latent space with losses funct.
    Parameters:
    - in_channel (int): Number of input channels. Default is 3 (RGB).
    - out_channel (int): Number of output channels. Default is 3 (RGB).
    - base_channel (int): Base number of channels for the network. Default is 64.
    - bitrate_conditional (bool): Whether to use bitrate conditional or not. Default is False.
    - channel_multiplier (list): List of multipliers for each layer. Default is [1, 2, 3, 4, 4], meaning the nummber of channel per layer will be increased by those multiplier.
    - hyperprior_channel_multiplier (list): List of multipliers for hyperprior channels. Default is [3, 3, 3], meaning the nummber of channel per layer will be increased by those multiplier.
    """

    """
    Định nghĩa lớp cơ bản thực hiện việc nén và giải nén đầu vào từ/đến không gian tiềm ẩn với các hàm mất mát.
    Tham số:
    - in_channel (int): Số kênh đầu vào. Mặc định là 3 (RGB).
    - out_channel (int): Số kênh đầu ra. Mặc định là 3 (RGB).
    - base_channel (int): Số kênh cơ bản cho mạng. Mặc định là 64.
    - bitrate_conditional (bool): Có sử dụng điều kiện bitrate hay không. Mặc định là False.
    - channel_multiplier (list): Danh sách các hệ số nhân cho mỗi lớp. Mặc định là [1, 2, 3, 4, 4], có nghĩa là số kênh mỗi lớp sẽ tăng lên bởi các hệ số nhân đó.
    - hyperprior_channel_multiplier (list): Danh sách các hệ số nhân cho các kênh hyperprior. Mặc định là [3, 3, 3], có nghĩa là số kênh mỗi lớp sẽ tăng lên bởi các hệ số nhân đó.
    """

    def __init__(
        self, 
        in_channel=3, 
        out_channel=3, 
        base_channel=64, 
        bitrate_conditional=False, 
        channel_multiplier=[1, 2, 3, 4, 4], 
        hyperprior_channel_multiplier= [3, 3, 3]
    ):
        super().__init__()

        ## --- Lưu trữ các tham số cấu hình ---
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.base_channel = base_channel
        self.bitrate_conditional = bitrate_conditional # vbr flag

        ## --- Tính toán số kênh cho mạng chính (Encoder/Decoder) ---

        # 1. Số kênh đầu ra cho mỗi lớp của mạng Encoder (xuôi)
        # Bắt đầu với kênh đầu vào, sau đó nhân base_channel với các hệ số
        # Ví dụ: [3, 64*1, 64*2, 64*3, 64*3] = [3, 64, 128, 192, 192]
        self.encoder_channels = [in_channel]
        for multiplier in channel_multiplier:
            self.encoder_channels.append(base_channel * multiplier)

        # 2. Cặp (kênh vào, kênh ra) cho mỗi lớp của mạng Encoder
        # Ví dụ: [(3, 64), (64, 128), (128, 192), (192, 192)]
        self.encoder_in_out_pairs = []
        for i in range(len(self.encoder_channels) - 1):
            in_c = self.encoder_channels[i]
            out_c = self.encoder_channels[i+1]
            self.encoder_in_out_pairs.append((in_c, out_c))

        # 3. Số kênh đầu ra cho mỗi lớp của mạng Decoder (ngược)
        # Bắt đầu với kênh đầu ra cuối cùng, sau đó nhân base_channel với các hệ số
        # Cuối cùng đảo ngược toàn bộ danh sách
        # Ví dụ: Tính [3, 64*1, 64*2, 64*3, 64*3] = [3, 64, 128, 192, 192]
        # Sau đó đảo ngược: [192, 192, 128, 64, 3]
        decoder_channels_temp = [out_channel]
        for multiplier in channel_multiplier:
             decoder_channels_temp.append(base_channel * multiplier)
        self.decoder_channels = list(reversed(decoder_channels_temp))

        # 4. Cặp (kênh vào, kênh ra) cho mỗi lớp của mạng Decoder
        # Ví dụ: [(192, 192), (192, 128), (128, 64), (64, 3)]
        self.decoder_in_out_pairs = []
        for i in range(len(self.decoder_channels) - 1):
            in_c = self.decoder_channels[i]
            out_c = self.decoder_channels[i+1]
            self.decoder_in_out_pairs.append((in_c, out_c))

        ## --- Tính toán số kênh cho mạng Hyperprior (Encoder/Decoder) ---
        # Lấy số kênh cuối cùng của mạng encoder chính làm kênh đầu vào cho hyper encoder
        hyper_encoder_start_channel = self.encoder_channels[-1]

        # 5. Số kênh đầu ra cho mỗi lớp của mạng Hyper Encoder (xuôi)
        # Bắt đầu với kênh cuối của encoder chính, sau đó nhân base_channel với các hệ số hyper
        # Ví dụ: [192, 64*3, 64*3, 64*3] = [192, 192, 192, 192]
        self.hyper_encoder_channels = [hyper_encoder_start_channel]
        for multiplier in hyperprior_channel_multiplier:
            self.hyper_encoder_channels.append(base_channel * multiplier)

        # 6. Cặp (kênh vào, kênh ra) cho mỗi lớp của mạng Hyper Encoder
        # Ví dụ: [(192, 192), (192, 192), (192, 192)]
        self.hyper_encoder_in_out_pairs = []
        for i in range(len(self.hyper_encoder_channels) - 1):
            in_c = self.hyper_encoder_channels[i]
            out_c = self.hyper_encoder_channels[i+1]
            self.hyper_encoder_in_out_pairs.append((in_c, out_c))

        # 7. Số kênh đầu ra cho mỗi lớp của mạng Hyper Decoder (ngược)
        # Bắt đầu với gấp đôi kênh cuối của encoder chính, sau đó nhân base_channel với các hệ số hyper
        # Cuối cùng đảo ngược toàn bộ danh sách
        # Ví dụ: Tính [192*2, 64*3, 64*3, 64*3] = [384, 192, 192, 192]
        # Sau đó đảo ngược: [192, 192, 192, 384]
        hyper_decoder_channels_temp = [hyper_encoder_start_channel * 2]
        for multiplier in hyperprior_channel_multiplier:
            hyper_decoder_channels_temp.append(base_channel * multiplier)
        self.hyper_decoder_channels = list(reversed(hyper_decoder_channels_temp))

        # 8. Cặp (kênh vào, kênh ra) cho mỗi lớp của mạng Hyper Decoder
        # Ví dụ: [(192, 192), (192, 192), (192, 384)]
        self.hyper_decoder_in_out_pairs = []
        for i in range(len(self.hyper_decoder_channels) - 1):
            in_c = self.hyper_decoder_channels[i] 
            out_c = self.hyper_decoder_channels[i+1]
            self.hyper_decoder_in_out_pairs.append((in_c, out_c))

        ## --- Khởi tạo các thành phần khác ---
        # Lấy số kênh cuối cùng của hyper encoder để khởi tạo prior
        prior_dim = self.hyper_encoder_channels[-1]
        self.hyper_prior = HyperPrior(in_channels=prior_dim)

        # Khởi tạo mạng:
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.hyper_encoder = nn.ModuleList()
        self.hyper_decoder = nn.ModuleList()

        # Xây dựng mang
        self.build_network()


    def build_network(self):
        """
        Xây dựng các thành phần mạng chính: Encoder, Decoder, HyperEncoder, HyperDecoder.
        Sử dụng các cặp kênh vào/ra đã được tính toán trong __init__.
        """
        logger.info("--- Xây dựng Encoder ---")
        # --- Xây dựng Encoder ---
        # Lặp qua các cặp kênh vào/ra của encoder
        for idx, (dim_in, dim_out) in enumerate(self.encoder_in_out_pairs):
            logger.info(f"Encoder Block {idx}: {dim_in} -> {dim_out}")
            # Tạo một khối gồm ResnetBlock, VariableBitrateCondition (tùy chọn), Downsample
            block = nn.ModuleList([
                # group_norm=True chỉ cho lớp đầu tiên (ind == 0).
                ResnetBlock(dim_in, dim_out, large_kernel=True if idx==0 else False),

                # VariableBitrateCondition: Nếu bitrate_conditional=True, thêm lớp điều kiện bitrate.
                # Nếu không, thêm lớp Identity (không làm gì cả).
                VariableBitrateCondition(1, dim_out) if self.bitrate_conditional else nn.Identity(),

                # Downsample: Giảm kích thước không gian của feature map.
                DownSampling(dim_out, dim_out)
            ])
            self.encoder.append(block)

        logger.info("--- Xây dựng Decoder ---")
        # --- Xây dựng Decoder ---
        # Lặp qua các cặp kênh vào/ra của decoder (đã đảo ngược)
        num_decoder_blocks = len(self.decoder_in_out_pairs)
        for idx, (dim_in, dim_out) in enumerate(self.decoder_in_out_pairs):
            is_last = (idx >= num_decoder_blocks - 1) # Kiểm tra xem có phải khối cuối cùng không
            logger.info(f"Decoder Block {idx}: {dim_in} -> {dim_out} (is_last={is_last})")
            resnet_vbr_out_dim = dim_in if is_last else dim_out

            # Tạo một khối gồm Upsample, ResnetBlock, VariableBitrateCondition (tùy chọn)
            # Thứ tự khác với encoder vì Upsample thường đứng trước Resnet trong decoder
            block = nn.ModuleList([
                UpSampling(dim_in, dim_in),
                ResnetBlock(dim_in, resnet_vbr_out_dim),
                VariableBitrateCondition(1, resnet_vbr_out_dim) if self.bitrate_conditional else nn.Identity(),
            ])
            self.decoder.append(block)


        logger.info("--- Xây dựng Hyper Encoder ---")
        # --- Xây dựng Hyper Encoder ---
        # Lặp qua các cặp kênh vào/ra của hyper encoder
        downscale_shape = []
        num_hyper_enc_blocks = len(self.hyper_encoder_in_out_pairs)
        for idx, (dim_in, dim_out) in enumerate(self.hyper_encoder_in_out_pairs):
            is_last = (idx >= num_hyper_enc_blocks - 1) # Kiểm tra khối cuối
            logger.info(f"HyperEncoder Block {idx}: {dim_in} -> {dim_out} (is_last={is_last})")
            block = nn.ModuleList([
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1) if idx == 0 else nn.Conv2d(dim_in, dim_out, kernel_size=5, stride=2, padding=2),
                # VariableBitrateCondition: Thêm nếu bitrate_conditional=True và *không phải* lớp cuối.
                VariableBitrateCondition(1, dim_out) if (self.bitrate_conditional and not is_last) else nn.Identity(),
                nn.LeakyReLU(0.2) if not is_last else nn.Identity()
            ])
            self.hyper_encoder.append(block)

        logger.info("--- Xây dựng Hyper Decoder ---")
        # --- Xây dựng Hyper Decoder --- 
        # Lặp qua các cặp kênh vào/ra của hyper decoder (đã đảo ngược)
        num_hyper_dec_blocks = len(self.hyper_decoder_in_out_pairs)
        for idx, (dim_in, dim_out) in enumerate(self.hyper_decoder_in_out_pairs):
            is_last = (idx >= num_hyper_dec_blocks - 1) # Kiểm tra khối cuối
            logger.info(f"HyperDecoder Block {idx}: {dim_in} -> {dim_out} (is_last={is_last})")

             # Tạo một khối gồm ConvTranspose2d/Conv2d, VariableBitrateCondition (tùy chọn), LeakyReLU (tùy chọn)
            block = nn.ModuleList([
                nn.ConvTranspose2d(dim_in, dim_out, kernel_size=5, stride=2, padding=2, output_padding=1) if not is_last \
                else nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                VariableBitrateCondition(1, dim_out) if (self.bitrate_conditional and not is_last) else nn.Identity(),
                nn.LeakyReLU(0.2) if not is_last else nn.Identity()
            ])
            self.hyper_decoder.append(block)


    def encode(self, input_image, bitrate_condition=None):
        """
        Thực hiện quá trình mã hóa ảnh đầu vào thành biểu diễn ẩn chính (latent)
        và biểu diễn ẩn siêu tiên nghiệm (hyper-latent).

        Args:
            input_image (Tensor): Tensor ảnh đầu vào (N, C, H, W).
            bitrate_condition (Tensor, optional): Embedding điều kiện bitrate (nếu có).

        Returns:
            tuple: Chứa:
                - q_latent (Tensor): Biểu diễn ẩn chính đã lượng tử hóa.
                - q_hyper_latent (Tensor): Biểu diễn ẩn siêu tiên nghiệm đã lượng tử hóa.
                - state4bpp (dict): Dictionary chứa các tensor trung gian cần thiết để tính toán BPP (latent, hyper_latent, latent_distribution).
        """
        logger.info("--- Bắt đầu Encode ---")
        current_features = input_image

        # 1. Mạng mã hóa chính (Encoder)
        logger.info("Chạy qua Encoder chính...")
        # self.enc là ModuleList của các ModuleList con [ResnetBlock, VBR/Identity, Downsample]
        for i, encoder_block in enumerate(self.encoder):
            resnet_layer = encoder_block[0]
            vbr_layer = encoder_block[1] # VBRCondition hoặc nn.Identity
            downsample_layer = encoder_block[2]

            current_features = resnet_layer(current_features)
            # Áp dụng điều kiện bitrate nếu được kích hoạt
            if self.bitrate_conditional:
                current_features = vbr_layer(current_features, bitrate_condition)
            current_features = downsample_layer(current_features)

        # Lưu trữ biểu diễn ẩn chính trước khi lượng tử hóa
        latent = current_features
        logger.info(f"Latent shape: {latent.shape}")

        # 2. Mạng mã hóa siêu tiên nghiệm (Hyper Encoder)
        logger.info("Chạy qua Hyper Encoder...")
        hyper_encoder_input = latent # Đầu vào là latent từ encoder chính
        # self.hyper_enc là ModuleList của các ModuleList con [Conv, VBR/Identity, Act/Identity]
        num_hyper_enc_layers = len(self.hyper_encoder)
        downscale_shape = [hyper_encoder_input.shape] # Lưu trữ kích thước đầu vào cho mỗi khối
        for i, hyper_enc_block in enumerate(self.hyper_encoder):
            conv_layer = hyper_enc_block[0]
            vbr_layer = hyper_enc_block[1]
            activation_layer = hyper_enc_block[2]

            # Lưu lại kích thước sau khi downscale
            downscale_shape.append(hyper_encoder_input.shape) # Lưu trữ kích thước đầu vào cho mỗi khối

            hyper_encoder_input = conv_layer(hyper_encoder_input)
            # Áp dụng VBR, trừ lớp cuối cùng
            if self.bitrate_conditional and i < (num_hyper_enc_layers - 1):
                hyper_encoder_input = vbr_layer(hyper_encoder_input, bitrate_condition)
            hyper_encoder_input = activation_layer(hyper_encoder_input)
            logger.info(f"HyperEncoder block {i} output shape: {hyper_encoder_input.shape}")

        # Lưu trữ biểu diễn ẩn siêu tiên nghiệm trước khi lượng tử hóa
        hyper_latent = hyper_encoder_input
        logger.info(f"Hyper latent shape: {hyper_latent.shape}")

        # 3. Lượng tử hóa Hyper-Latent (sử dụng trong Hyper Decoder và tính BPP)
        # Sử dụng chế độ 'dequantize' (làm tròn) và median từ prior
        # Trong quá trình huấn luyện, hàm bpp sẽ lượng tử hóa lại với 'noise'
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.hyper_prior.medians)
        logger.info(f"Quantized Hyper latent shape: {q_hyper_latent.shape}")


        # 4. Mạng giải mã siêu tiên nghiệm (Hyper Decoder)
        logger.info("Chạy qua Hyper Decoder...")
        hyper_decoder_input = q_hyper_latent # Đầu vào là hyper-latent đã lượng tử hóa
        # self.hyper_dec là ModuleList của các ModuleList con [ConvT/Conv, VBR/Identity, Act/Identity]
        num_hyper_dec_layers = len(self.hyper_decoder)
        for i, hyper_dec_block in enumerate(self.hyper_decoder):

            deconv_layer = hyper_dec_block[0]
            vbr_layer = hyper_dec_block[1]
            activation_layer = hyper_dec_block[2]

            hyper_decoder_input = deconv_layer(hyper_decoder_input)
             # Áp dụng VBR, trừ lớp cuối cùng
            if self.bitrate_conditional and i < (num_hyper_dec_layers - 1):
                hyper_decoder_input = vbr_layer(hyper_decoder_input, bitrate_condition)
            hyper_decoder_input = activation_layer(hyper_decoder_input)

            # Đảm bảo quá trình upscale không gây lỗi mismatch
            if hyper_decoder_input.shape != downscale_shape[-(i+1)]:
                hyper_decoder_input = nn.functional.interpolate(hyper_decoder_input, size=downscale_shape[-(i+1)][2:], mode='bilinear', align_corners=False)

            logger.info(f"HyperDecoder block {i} output shape: {hyper_decoder_input.shape}")
        # Đầu ra của hyper-decoder chứa tham số (mean, scale) cho phân phối của latent
        hyper_decoder_output = hyper_decoder_input
        logger.info(f"Hyper Decoder output shape: {hyper_decoder_output.shape}")

        # 5. Tạo phân phối cho Latent và Lượng tử hóa Latent
        # Chia đầu ra hyper-decoder thành mean và scale dọc theo chiều kênh
        mean, scale = hyper_decoder_output.chunk(2, dim=1)
        # Đảm bảo scale không quá nhỏ (ổn định số học)
        scale = scale.clamp(min=0.1) # Giá trị clamp có thể cần điều chỉnh
        logger.info(f"Latent distribution params shapes: mean={mean.shape}, scale={scale.shape}")

        # Tạo đối tượng phân phối Gaussian
        latent_distribution = NormalDistribution(mean, scale)

        # Lượng tử hóa latent chính (sử dụng mean của phân phối làm tâm)
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        logger.info(f"Quantized Latent shape: {q_latent.shape}")


        # 6. Chuẩn bị đầu ra
        # Lưu các trạng thái cần thiết để tính BPP sau này
        state4bpp = {
            "latent": latent,                 # Latent gốc (trước quantization)
            "hyper_latent": hyper_latent,     # Hyper-latent gốc (trước quantization)
            "latent_distribution": latent_distribution, # Đối tượng phân phối của latent
        }
        logger.info("--- Kết thúc Encode ---")
        return q_latent, q_hyper_latent, state4bpp


    def decode(self, q_latent, bitrate_condition=None):
        """
        Thực hiện quá trình giải mã từ biểu diễn ẩn chính đã lượng tử hóa.

        Lưu ý: Logic gốc của hàm này trả về một danh sách các feature map trung gian
        thay vì một ảnh tái tạo cuối cùng duy nhất. Hành vi này được giữ lại.
        Việc sử dụng các lớp trong khối cũng được điều chỉnh để khớp với cách
        chúng được xây dựng trong `build_network`.

        Args:
            q_latent (Tensor): Biểu diễn ẩn chính đã lượng tử hóa (đầu ra từ encode).
            bitrate_condition (Tensor, optional): Embedding điều kiện bitrate (nếu có).

        Returns:
            list[Tensor]: Danh sách các tensor đầu ra từ mỗi khối của decoder,
                          được đảo ngược thứ tự.
        """
        logger.info("--- Bắt đầu Decode ---")
        intermediate_outputs = []
        current_features = q_latent # Bắt đầu với latent đã lượng tử hóa
        logger.info(f"Decoder input shape: {current_features.shape}")

        # self.dec là ModuleList của các ModuleList con [Upsample, ResnetBlock, VBR/Identity]
        for i, decoder_block in enumerate(self.decoder):
            upsample_layer = decoder_block[0]
            resnet_layer = decoder_block[1]
            vbr_layer = decoder_block[2] # VBRCondition hoặc nn.Identity

            current_features = upsample_layer(current_features)

            current_features = resnet_layer(current_features)

            # Áp dụng điều kiện bitrate nếu được kích hoạt
            if self.bitrate_conditional:
                current_features = vbr_layer(current_features, bitrate_condition)

            # Lưu lại đầu ra của khối này (theo logic gốc)
            intermediate_outputs.append(current_features)
            logger.info(f"Decoder block {i} final output shape: {current_features.shape}")

        logger.info("--- Kết thúc Decode ---")
        # Trả về danh sách các đầu ra đã đảo ngược (theo logic gốc)
        return intermediate_outputs[::-1]

    def bpp(self, input_shape, state4bpp):
        """
        Tính toán số bit trên mỗi pixel (BPP) ước lượng.

        Args:
            input_shape (torch.Size): Kích thước của ảnh đầu vào gốc (để lấy H, W).
            state4bpp (dict): Dictionary chứa các tensor trung gian từ hàm encode.

        Returns:
            Tensor: Giá trị BPP ước lượng (thường là tensor 1 chiều, mỗi phần tử cho 1 ảnh trong batch).
        """
        logger.info("--- Bắt đầu tính BPP ---")
        B, _, H, W = input_shape # Lấy kích thước ảnh gốc
        # Lấy các tensor cần thiết từ state4bpp
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        logger.info(f"Input shape for BPP: B={B}, H={H}, W={W}")
        logger.info(f"Latent shape: {latent.shape}, Hyper latent shape: {hyper_latent.shape}")


        # Lượng tử hóa lại (khác nhau giữa training và evaluation)
        if self.training:
            # Khi training, thêm nhiễu lượng tử hóa để huấn luyện mô hình entropy
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
            logger.info("BPP calculation in training mode (using noise quantization)")
        else:
            # Khi evaluation, sử dụng làm tròn (dequantize) giống như trong encode
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.hyper_prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
            logger.info("BPP calculation in eval mode (using dequantize)")

        # Tính rate (số bit) từ likelihood
        # Rate = -log2(Likelihood)
        # Sử dụng likelihood từ HyperPrior cho q_hyper_latent
        hyper_rate = -self.hyper_prior.likelihood(q_hyper_latent).log2()
        # Sử dụng likelihood từ NormalDistribution (tham số hóa bởi hyper-decoder) cho q_latent
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        logger.info(f"Hyper rate shape: {hyper_rate.shape}, Cond rate shape: {cond_rate.shape}")


        # Tính tổng số bit cho mỗi ảnh trong batch
        total_bits = hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))
        logger.info(f"Total bits per image shape: {total_bits.detach().cpu().numpy()}")
        # Tính BPP bằng cách chia tổng số bit cho số pixel
        num_pixels = H * W
        bpp_estimate = total_bits / num_pixels
        logger.info(f"Calculated BPP shape: {bpp_estimate.shape}")
        logger.info("--- Kết thúc tính BPP ---")
        return bpp_estimate


    def forward(self, input_image, bitrate_condition=None):
        """
        Hàm forward chính của mô hình Compressor.

        Thực hiện:
        1. Mã hóa ảnh đầu vào (`self.encode`).
        2. Tính toán BPP ước lượng (`self.bpp`).
        3. Giải mã từ latent đã lượng tử hóa (`self.decode`).
        4. Trả về kết quả dưới dạng dictionary.

        Args:
            input_image (Tensor): Tensor ảnh đầu vào (N, C, H, W).
            bitrate_condition (Tensor, optional): Embedding điều kiện bitrate (nếu có).

        Returns:
            dict: Dictionary chứa các kết quả:
                - "output": Kết quả từ decoder (list các tensor theo logic gốc).
                - "bpp": Giá trị BPP ước lượng.
                - "q_latent": Biểu diễn ẩn chính đã lượng tử hóa.
                - "q_hyper_latent": Biểu diễn ẩn siêu tiên nghiệm đã lượng tử hóa.
        """
        logger.info("--- Bắt đầu Forward ---")
        # 1. Mã hóa
        logger.critical(f"input image shape: {input_image.shape}")
        q_latent, q_hyper_latent, state4bpp = self.encode(input_image, bitrate_condition)

        # 2. Tính BPP
        bpp_estimate = self.bpp(input_image.shape, state4bpp)

        # 3. Giải mã
        # Đầu vào của decode là latent đã lượng tử hóa
        decoded_output = self.decode(q_latent, bitrate_condition)

        logger.info("--- Kết thúc Forward ---")
        # 4. Trả về kết quả
        return {
            "output": decoded_output,      # Kết quả từ decoder
            "bpp": bpp_estimate,           # Giá trị BPP
            "q_latent": q_latent,          # Latent lượng tử hóa
            "q_hyper_latent": q_hyper_latent, # Hyper-latent lượng tử hóa
            # Bạn có thể thêm các giá trị khác vào đây nếu cần, ví dụ:
            # "state4bpp": state4bpp # Nếu cần truy cập bên ngoài
        }

        

    def prior_probability_loss(self):
        return self.hyper_prior.get_extraloss()

        
# Testing the Compressor class
compressor = Compressor(
    in_channel=3,
    out_channel=3,
    base_channel=64,
    channel_multiplier=[1, 2, 3, 3],
    hyperprior_channel_multiplier=[3, 3, 3]
)

try:
    image = 'dogpicture.png'
    # Load the image and convert it to a tensor
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1200, 600)),  # Resize to (256, 256) for testing
    ])

    img = Image.open(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension (N, C, H, W)

    dump_tensor = img_tensor
    output = compressor(dump_tensor, bitrate_condition=None)
    for key, value in output.items():
        logger.info(f"{key}: {value.shape if isinstance(value, torch.Tensor) else len(value)}")

    logger.critical("Testing completed successfully.")
except Exception as e:
    logger.error(f"Error during testing: {e}")
    raise e