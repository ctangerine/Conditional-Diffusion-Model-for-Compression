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
    """
    Làm tròn tensor `input` dựa trên một tâm (offset) `loc`.
    Nó làm tròn hiệu số `input - loc` và sau đó cộng lại `loc`.
    Sử dụng STERound để cho phép gradient đi qua.

    Args:
        input (Tensor): Tensor cần làm tròn.
        loc (Tensor): Tensor chứa giá trị tâm (offset) để làm tròn quanh nó.
                      Thường là mean hoặc median của phân phối ước lượng.

    Returns:
        Tensor: Tensor đã được làm tròn quanh `loc`.
    """
    # Tính hiệu số so với tâm
    difference = input - loc
    # Làm tròn hiệu số (với STE)
    rounded_difference = STERound.apply(difference)
    # Cộng lại tâm để có giá trị làm tròn cuối cùng
    return rounded_difference + loc

def quantize(x, mode='noise', offset=None):
    """
    Thực hiện các phương pháp lượng tử hóa/làm tròn khác nhau dựa trên `mode`.

    Args:
        x (Tensor): Tensor đầu vào cần lượng tử hóa.
        mode (str): Chế độ lượng tử hóa:
                    - 'noise': Thêm nhiễu lượng tử hóa (thường dùng khi training).
                    - 'round': Làm tròn tới số nguyên gần nhất (sử dụng STE).
                    - 'dequantize': Làm tròn quanh một `offset` (sử dụng STE).
                                     Tên 'dequantize' ở đây có thể hơi gây hiểu nhầm,
                                     nó thực chất là làm tròn có tâm.
        offset (Tensor, optional): Tâm làm tròn, cần thiết cho mode 'dequantize'.

    Returns:
        Tensor: Tensor đã được xử lý theo `mode` đã chọn.

    Raises:
        NotImplementedError: Nếu `mode` không được hỗ trợ.
    """
    if mode == 'noise':
        # Thêm nhiễu đồng phục [-0.5, 0.5)
        return noise(x, scale=1.0)
    elif mode == 'round':
        # Làm tròn tới số nguyên gần nhất, cho phép gradient đi qua
        return STERound.apply(x)
    elif mode == 'dequantize':
        # Làm tròn quanh offset, cho phép gradient đi qua
        if offset is None:
            raise ValueError("Offset must be provided for 'dequantize' mode.")
        return round_w_offset(x, offset)
    else:
        raise NotImplementedError(f"Quantization mode '{mode}' not implemented.")


class STERound(Function):
    """
    Hàm autograd tùy chỉnh cho Straight-Through Estimator (STE) của phép làm tròn.

    Forward pass: Thực hiện phép làm tròn chuẩn (x.round()).
    Backward pass: Coi phép làm tròn như một hàm đồng nhất (identity),
                   cho phép gradient đi qua không thay đổi.
    Điều này cho phép huấn luyện mạng nơ-ron có các bước lượng tử hóa/làm tròn rời rạc.
    """
    @staticmethod
    def forward(ctx, x):
        """Thực hiện làm tròn trong forward pass."""
        # ctx là context để lưu thông tin cho backward pass (không cần ở đây)
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        """Truyền gradient qua không đổi trong backward pass."""
        # Trả về gradient cho đầu vào x là chính grad_output
        return grad_output

# ==============================================================================
# ===== Hàm Autograd chặn dưới/trên (Bounding Autograd Functions) =====
# ==============================================================================

class LowerBound(Function):
    """
    Hàm autograd tùy chỉnh để áp dụng cận dưới cho một tensor.

    Forward pass: Trả về torch.max(inputs, bound). Đảm bảo giá trị không nhỏ hơn `bound`.
    Backward pass: Cho phép gradient đi qua (`grad_output`) nếu đầu vào gốc (`inputs`)
                   lớn hơn hoặc bằng `bound`, HOẶC nếu gradient đầu ra (`grad_output`)
                   là âm. Logic backward này dựa trên Ballé et al. 2018 (Appendix 1.1)
                   để ổn định việc huấn luyện các mô hình likelihood.
    """
    @staticmethod
    def forward(ctx, inputs, bound):
        """Áp dụng cận dưới trong forward pass."""
        # Tạo tensor cận dưới có cùng shape và device với inputs
        # Sử dụng torch.as_tensor để xử lý device và dtype
        bound_tensor = torch.as_tensor(bound, dtype=inputs.dtype, device=inputs.device)
        # Lưu inputs và bound_tensor để dùng trong backward
        ctx.save_for_backward(inputs, bound_tensor)
        # Trả về giá trị lớn nhất giữa inputs và bound_tensor
        return torch.max(inputs, bound_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        """Tính gradient tùy chỉnh trong backward pass."""
        inputs, bound_tensor = ctx.saved_tensors # Lấy lại các tensor đã lưu

        # Điều kiện 1: Gradient đi qua nếu input gốc >= bound
        pass_through_if_input_ge_bound = inputs >= bound_tensor
        # Điều kiện 2: Gradient đi qua nếu gradient từ lớp sau < 0
        pass_through_if_grad_lt_zero = grad_output < 0

        # Kết hợp hai điều kiện: gradient đi qua nếu 1 trong 2 đúng
        pass_through_mask = pass_through_if_input_ge_bound | pass_through_if_grad_lt_zero

        # Nhân grad_output với mặt nạ boolean (đã chuyển sang float)
        # Trả về gradient cho inputs và None cho bound (vì bound không cần gradient)
        return pass_through_mask.type(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    """
    Hàm autograd tùy chỉnh để áp dụng cận trên cho một tensor.

    Forward pass: Trả về torch.min(inputs, bound). Đảm bảo giá trị không lớn hơn `bound`.
    Backward pass: Logic đối xứng với LowerBound. Cho phép gradient đi qua nếu
                   `inputs <= bound` HOẶC `grad_output > 0`.
    """
    @staticmethod
    def forward(ctx, inputs, bound):
        """Áp dụng cận trên trong forward pass."""
        bound_tensor = torch.as_tensor(bound, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, bound_tensor)
        return torch.min(inputs, bound_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        """Tính gradient tùy chỉnh trong backward pass."""
        inputs, bound_tensor = ctx.saved_tensors

        # Điều kiện 1: Gradient đi qua nếu input gốc <= bound
        pass_through_if_input_le_bound = inputs <= bound_tensor
        # Điều kiện 2: Gradient đi qua nếu gradient từ lớp sau > 0
        pass_through_if_grad_gt_zero = grad_output > 0

        # Kết hợp hai điều kiện
        pass_through_mask = pass_through_if_input_le_bound | pass_through_if_grad_gt_zero

        # Nhân grad_output với mặt nạ
        return pass_through_mask.type(grad_output.dtype) * grad_output, None

# ==============================================================================
# ===== Lớp Phân Phối Chuẩn (Normal Distribution Class) =====
# ==============================================================================

class NormalDistribution:
    """
    Đại diện cho một phân phối Gaussian (Normal) đa biến,
    với giả định các chiều là độc lập.
    Được thiết kế đặc biệt cho các mô hình nén, cung cấp phương thức tính
    likelihood của giá trị lượng tử hóa.
    """
    def __init__(self, loc, scale):
        """
        Khởi tạo phân phối chuẩn.

        Args:
            loc (Tensor): Tensor chứa giá trị trung bình (mean) của phân phối.
            scale (Tensor): Tensor chứa độ lệch chuẩn (standard deviation) của phân phối.
                           Phải có cùng shape với `loc`.
        """
        # Đảm bảo loc và scale có cùng shape
        assert loc.shape == scale.shape, f"Shape mismatch: loc {loc.shape} vs scale {scale.shape}"
        self.loc = loc     # Trung bình
        self.scale = scale # Độ lệch chuẩn

    @property
    def mean(self):
        """
        Trả về giá trị trung bình (loc) của phân phối, được detach khỏi đồ thị tính toán.
        """
        return self.loc.detach()

    def std_cdf(self, inputs):
        """
        Tính toán Hàm Phân Phối Tích Lũy (CDF) của phân phối chuẩn *chuẩn hóa* (mean=0, std=1).
        Sử dụng hàm lỗi bổ sung (erfc) để tính toán ổn định.
        CDF(x) = 0.5 * erfc(-x / sqrt(2))

        Args:
            inputs (Tensor): Giá trị đầu vào (đã được chuẩn hóa: (x - mean) / scale).

        Returns:
            Tensor: Giá trị CDF của phân phối chuẩn chuẩn hóa tại `inputs`.
        """
        half = 0.5
        # Sử dụng hằng số -(2**-0.5) = -1/sqrt(2)
        const = -(2**-0.5)
        return half * torch.erfc(const * inputs)

    def sample(self):
        """
        Tạo mẫu ngẫu nhiên từ phân phối chuẩn này.

        Returns:
            Tensor: Mẫu ngẫu nhiên có cùng shape với `loc` và `scale`.
        """
        # Tạo nhiễu chuẩn chuẩn hóa và scale/shift theo tham số phân phối
        return self.scale * torch.randn_like(self.scale) + self.loc

    def likelihood(self, x, min_likelihood=1e-9):
        """
        Tính toán likelihood (khả năng xảy ra) của giá trị lượng tử hóa `x`.
        Xấp xỉ P(X = x) bằng P(x - 0.5 < X <= x + 0.5),
        tức là CDF(x + 0.5) - CDF(x - 0.5).

        Args:
            x (Tensor): Giá trị lượng tử hóa (thường là số nguyên).
            min_likelihood (float): Cận dưới cho likelihood để tránh log(0).

        Returns:
            Tensor: Likelihood của `x` theo phân phối này.
        """
        # Chuẩn hóa các điểm biên của khoảng lượng tử hóa: (value - mean) / scale
        standardized_upper = (x + 0.5 - self.loc) / self.scale
        standardized_lower = (x - 0.5 - self.loc) / self.scale

        # Tính CDF tại các điểm biên đã chuẩn hóa
        upper_cdf = self.std_cdf(standardized_upper)
        lower_cdf = self.std_cdf(standardized_lower)

        # Likelihood là hiệu của CDF
        likelihood_val = upper_cdf - lower_cdf

        # Áp dụng cận dưới bằng LowerBound để đảm bảo gradient đúng và tránh giá trị 0
        return LowerBound.apply(likelihood_val, min_likelihood)

    def scaled_likelihood(self, x, s=1.0, min_likelihood=1e-9):
        """
        Tính toán likelihood trên một khoảng có độ rộng `s` thay vì 1.0.
        Xấp xỉ P(x - s/2 < X <= x + s/2) = CDF(x + s/2) - CDF(x - s/2).

        Args:
            x (Tensor): Giá trị lượng tử hóa hoặc tâm khoảng.
            s (float): Độ rộng của khoảng tính likelihood.
            min_likelihood (float): Cận dưới cho likelihood.

        Returns:
            Tensor: Likelihood của khoảng [x - s/2, x + s/2] theo phân phối này.
        """
        half_s = s * 0.5
        # Chuẩn hóa các điểm biên của khoảng [-s/2, s/2] quanh x
        standardized_upper = (x + half_s - self.loc) / self.scale
        standardized_lower = (x - half_s - self.loc) / self.scale

        # Tính CDF
        upper_cdf = self.std_cdf(standardized_upper)
        lower_cdf = self.std_cdf(standardized_lower)

        # Tính likelihood và áp dụng cận dưới
        likelihood_val = upper_cdf - lower_cdf
        return LowerBound.apply(likelihood_val, min_likelihood)