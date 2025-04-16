import torch
import torch.nn as nn

from network_components.hyper_prior import HyperPrior

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

    def prior_probability_loss(self):
        return self.hyper_prior.get_extraloss()

        
# Testing the Compressor class
# compressor = Compressor(
#     in_channel=3,
#     out_channel=3,
#     base_channel=64,
#     channel_multiplier=[1, 2, 3, 3],
#     hyperprior_channel_multiplier=[3, 3, 3]
# )

# print("--- Kênh Encoder ---")
# print("Kênh:", compressor.encoder_channels)
# print("Cặp In/Out:", compressor.encoder_in_out_pairs)

# print("\n--- Kênh Decoder ---")
# print("Kênh:", compressor.decoder_channels)
# print("Cặp In/Out:", compressor.decoder_in_out_pairs)

# print("\n--- Kênh Hyper Encoder ---")
# print("Kênh:", compressor.hyper_encoder_channels)
# print("Cặp In/Out:", compressor.hyper_encoder_in_out_pairs)

# print("\n--- Kênh Hyper Decoder ---")
# print("Kênh:", compressor.hyper_decoder_channels)
# print("Cặp In/Out:", compressor.hyper_decoder_in_out_pairs)

# print(f"\nBitrate Conditional (VBR): {compressor.bitrate_conditional}")
# print(f"Prior: {compressor.prior}")