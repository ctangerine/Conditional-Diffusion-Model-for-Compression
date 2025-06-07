import torch
import numpy as np
from torch.autograd import Function

def extract(a, t, x_shape):
    """
    Trích xuất các giá trị từ tensor `a` tại các chỉ số `t` và reshape kết quả
    để phù hợp với việc broadcasting với tensor có shape `x_shape`.
    Thường dùng trong diffusion models để lấy các giá trị alpha, beta,...
    tương ứng với timestep `t` và áp dụng chúng lên tensor `x`.

    Args:
        a (Tensor): Tensor nguồn (thường là 1D) chứa các giá trị cần trích xuất (ví dụ: alphas_cumprod).
        t (Tensor): Tensor chứa các chỉ số (timesteps) để lấy giá trị từ `a`.
        x_shape (tuple): Shape của tensor mục tiêu (ví dụ: ảnh) mà kết quả cần broadcast tới.

    Returns:
        Tensor: Tensor chứa các giá trị đã trích xuất từ `a` tại chỉ số `t`,
                được reshape thành (batch_size, 1, 1, ...) để khớp với `x_shape`.
    """
    batch_size = t.shape[0]
    # a.gather(-1, t) lấy các giá trị trong a tại chỉ số t dọc theo chiều cuối cùng
    out = a.gather(-1, t)
    # reshape thành (batch_size, 1, 1, ...) để broadcast
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def extract_tensor(a, t):
    """
    Trích xuất các phần tử từ tensor `a` sử dụng `t` làm chỉ số cho chiều thứ nhất
    và một dãy số từ 0 đến len(t)-1 làm chỉ số cho chiều thứ hai.
    Hành vi này khá cụ thể, thường dùng khi `a` là 2D và bạn muốn lấy một
    "đường chéo" được xác định bởi `t`.

    Args:
        a (Tensor): Tensor nguồn (thường là 2D hoặc cao hơn).
        t (Tensor): Tensor 1D chứa các chỉ số cho chiều đầu tiên của `a`.

    Returns:
        Tensor: Tensor 1D chứa các phần tử a[t[0], 0], a[t[1], 1], ...
    """
    # Lấy các phần tử a[t[i], i] cho mỗi i trong range(len(t))
    return a[t, torch.arange(len(t), device=t.device)] # Đảm bảo arange cùng device với t

def noise_like(shape, device, repeat=False):
    """
    Tạo một tensor nhiễu Gaussian (phân phối chuẩn) có shape được chỉ định.
    Có tùy chọn lặp lại nhiễu của mẫu đầu tiên cho toàn bộ batch.

    Args:
        shape (tuple): Shape của tensor nhiễu cần tạo.
        device: Thiết bị (cpu/cuda) để tạo tensor trên đó.
        repeat (bool): Nếu True, tạo nhiễu cho mẫu đầu tiên (shape[0]=1)
                       và lặp lại nó cho toàn bộ batch size (shape[0]).
                       Nếu False, tạo nhiễu độc lập cho mỗi mẫu trong batch.

    Returns:
        Tensor: Tensor nhiễu Gaussian.
    """
    if repeat:
        # Tạo nhiễu cho batch size = 1, giữ nguyên các chiều còn lại
        noise_sample = torch.randn((1, *shape[1:]), device=device)
        # Lặp lại nhiễu này shape[0] lần dọc theo chiều batch
        return noise_sample.repeat(shape[0], *((1,) * (len(shape) - 1)))
    else:
        # Tạo nhiễu chuẩn với shape đầy đủ
        return torch.randn(shape, device=device)

# ==============================================================================
# ===== Lịch trình Beta (Diffusion Beta Schedules) =====
# ==============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Tạo lịch trình beta dựa trên hàm cosine, như đề xuất trong bài báo
    "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).
    Lịch trình này thường dẫn đến kết quả tốt hơn lịch trình tuyến tính.

    Args:
        timesteps (int): Tổng số bước thời gian (khuếch tán).
        s (float): Hằng số offset nhỏ để ngăn beta quá nhỏ ở đầu lịch trình.

    Returns:
        np.ndarray: Mảng NumPy 1D chứa các giá trị beta cho mỗi timestep.
    """
    steps = timesteps + 1
    # Tạo chuỗi giá trị x từ 0 đến steps
    x = np.linspace(0, steps, steps, dtype=np.float64)
    # Tính alphas_cumprod theo công thức cosine schedule
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    # Chuẩn hóa để giá trị đầu tiên là 1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # Tính betas từ alphas_cumprod: beta_t = 1 - alpha_t = 1 - alphas_cumprod_t / alphas_cumprod_{t-1}
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Cắt giá trị betas để đảm bảo nằm trong khoảng [0, 0.999]
    return np.clip(betas, a_min=0., a_max=0.999)

def linear_beta_schedule(timesteps):
    """
    Tạo lịch trình beta tuyến tính đơn giản, tăng dần từ beta_start đến beta_end.
    Đây là lịch trình được sử dụng trong bài báo gốc DDPM.

    Args:
        timesteps (int): Tổng số bước thời gian (khuếch tán).

    Returns:
        np.ndarray: Mảng NumPy 1D chứa các giá trị beta cho mỗi timestep.
    """
    # Scale các giá trị gốc (0.0001, 0.02 thường dùng cho 1000 bước) theo số timesteps
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    # Tạo mảng tuyến tính từ beta_start đến beta_end
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

# ==============================================================================
# ===== Lượng tử hóa và Làm tròn (Quantization & Rounding) =====
# ==============================================================================

def noise(input, scale):
    """
    Thêm nhiễu lượng tử hóa vào tensor đầu vào.
    Nhiễu được lấy từ phân phối đều trong khoảng [-0.5, 0.5] và nhân với scale.
    Thường dùng trong quá trình *huấn luyện* mô hình nén có lượng tử hóa.

    Args:
        input (Tensor): Tensor đầu vào.
        scale (float): Hệ số scale cho nhiễu. Thường là 1.0.

    Returns:
        Tensor: Tensor đầu vào cộng với nhiễu lượng tử hóa.
    """
    # torch.rand_like(input) tạo số ngẫu nhiên đều [0, 1)
    # Trừ 0.5 để dịch chuyển về [-0.5, 0.5)
    uniform_noise = torch.rand_like(input) - 0.5
    return input + scale * uniform_noise

def round_w_offset(input, loc):
    diff = STERound.apply(input - loc)
    return diff + loc


def quantize(x, mode='noise', offset=None):
    if mode == 'noise':
        return noise(x, 1)
    elif mode == 'round':
        return STERound.apply(x)
    elif mode == 'dequantize':
        return round_w_offset(x, offset)
    else:
        raise NotImplementedError



class STERound(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.min(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs <= b
        pass_through_2 = grad_output > 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class NormalDistribution:
    '''
        A normal distribution
    '''
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc.detach()

    def std_cdf(self, inputs):
        half = 0.5
        const = -(2**-0.5)
        return half * torch.erfc(const * inputs)

    def sample(self):
        return self.scale * torch.randn_like(self.scale) + self.loc

    def likelihood(self, x, min=1e-9):
        x = torch.abs(x - self.loc)
        upper = self.std_cdf((.5 - x) / self.scale)
        lower = self.std_cdf((-.5 - x) / self.scale)
        return LowerBound.apply(upper - lower, min)

    def scaled_likelihood(self, x, s=1, min=1e-9):
        x = torch.abs(x - self.loc)
        s = s * .5
        upper = self.std_cdf((s - x) / self.scale)
        lower = self.std_cdf((-s - x) / self.scale)
        return LowerBound.apply(upper - lower, min)
    

def check_valid(tensor):
    # Check if the tensor is valid (not NaN or Inf and not empty)
    if tensor is None or tensor.numel() == 0:
        return False
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    return True
