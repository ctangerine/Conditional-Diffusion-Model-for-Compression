import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_components.utils import LowerBound

class PriorFunction(nn.Module):
    """
    Description:
        Prior Function, a module that applies a linear transformation to the input tensor and adds a bias term.
        The linear transformation is applied by multiplying the input tensor with the weight tensor and adding the bias term.
    Args:
        parallel_dims: int, the number of parallel dimensions
        in_channels: int, the number of input channels
        out_channels: int, the number of output channels
        scale: float, the scale of the weight tensor
        bias: bool, if True, a bias term will be added to the output tensor
    Operations:
        1. Apply a linear transformation to the input tensor by multiplying the input tensor with the weight tensor
        2. Add a bias term to the output tensor if bias is True
        3. Return the output tensor
    """

    """
    Mô tả:
        Hàm Prior, một module áp dụng phép biến đổi tuyến tính cho tensor đầu vào và thêm một hạng tử bias.
        Phép biến đổi tuyến tính được thực hiện bằng cách nhân tensor đầu vào với tensor trọng số và thêm hạng tử bias.
    Tham số:
        parallel_dims: int, số chiều song song
        in_features: int, số kênh đầu vào (bây giờ là đặc trưng của image dưới dạng latent variable)
        out_features: int, số kênh đầu ra (dưới dạng phân phối tiền nghiệm của biến đặc trưng)
        scale: float, tỷ lệ của tensor trọng số
        bias: bool, nếu True, một hạng tử bias sẽ được thêm vào tensor đầu ra
    Thực thi:
        1. Áp dụng phép biến đổi tuyến tính cho tensor đầu vào bằng cách nhân tensor đầu vào với tensor trọng số
        2. Thêm một hạng tử bias vào tensor đầu ra nếu bias là True
        3. Trả về tensor đầu ra
    """

    __constants__ = ['bias', 'in_features', 'out_channels']

    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True, device=None):
        super(PriorFunction, self).__init__()
        self.parallel_dims = parallel_dims
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.bias = bias
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, 1, out_features))
        else:
            self.register_parameter('bias', None)

        self.weight = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, in_features, out_features))

        self.reset_parameters(scale)
        
        # Move parameters to device
        self.to(self.device)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, input, detach=False):
        """Applies the transformation: input @ softplus(weight) + bias"""
        # input shape expected: (parallel_dims, batch_size, ..., in_features)
        # weight shape:        (parallel_dims, 1, 1, in_features, out_features)
        # bias shape:          (parallel_dims, 1, 1, 1, out_features)
        
        # Move input to device
        input = input.to(self.device)
        
        weight_transformed = F.softplus(self.weight)

        if detach:
            # Sử dụng .detach() trên cả weight và bias nếu cần detach
            output = torch.matmul(input, weight_transformed.detach())
            if self.bias is not None:
                output = output + self.bias.detach()
            return output
        else:
            output = torch.matmul(input, weight_transformed)
            if self.bias is not None:
                output = output + self.bias
            return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class HyperPrior(nn.Module):
    """
    Mô tả:
        Mô hình Prior linh hoạt dựa trên Ballé et al. 2018 (Phụ lục 6.1).
        Nhiệm vụ chính là mô hình hóa phân phối xác suất của tensor biểu diễn ẩn 'y'
        trong các mô hình nén dữ liệu học sâu. Cung cấp các phương thức để tính
        likelihood (khả năng xảy ra) cần thiết cho quá trình mã hóa entropy.

    Tham số khởi tạo:
        in_channels (int): Số kênh của tensor biểu diễn ẩn 'y' (ví dụ: 256).
        transform_dims (list[int]): Danh sách các kích thước đặc trưng trung gian
                                    cho chuỗi các phép biến đổi bên trong (ví dụ: [3, 3, 3]).
        init_scale (float): Hệ số scale ban đầu dùng để khởi tạo tham số (ví dụ: 10.0).
        device (torch.device): Thiết bị để đặt mô hình (mặc định: None, sẽ sử dụng CUDA nếu có).
    """
    def __init__(self, in_channels=256, transform_dims=[3, 3, 3], init_scale=10.0, device=None):
        """
        Hàm khởi tạo của lớp HyperPrior.

        Thực hiện:
        1. Gọi hàm khởi tạo của lớp cha (nn.Module).
        2. Lưu số kênh đầu vào (`in_channels`).
        3. Xây dựng danh sách đầy đủ các kích thước biến đổi (`transform_dims_full`)
           bằng cách thêm 1 vào đầu và cuối `transform_dims` (ví dụ: [1, 3, 3, 3, 1]).
           Số 1 đại diện cho kích thước đầu vào/ra cuối cùng cho mỗi kênh khi tính CDF.
        4. Tính toán số lớp `PriorFunction` cần tạo (`num_chained_layers`).
        5. Tính hệ số scale cơ sở (`base_scale_factor`) để phân bổ `init_scale`
           cho các lớp con.
        6. Tạo một chuỗi các lớp `PriorFunction` (`self.affine_transforms`):
           - Lặp qua `num_chained_layers`.
           - Xác định `in_features`, `out_features` cho lớp `PriorFunction` hiện tại.
           - Tính `init_prior_scale` cụ thể cho lớp này dựa trên công thức trong bài báo.
           - Khởi tạo `PriorFunction` với các tham số phù hợp (parallel_dims=in_channels).
           - Lưu các lớp này vào `nn.ModuleList` để PyTorch quản lý tham số.
        7. Tạo các tham số 'a' (`self.nonlinear_factors`):
           - Các tham số này dùng cho phần phi tuyến tính trong hàm CDF.
           - Kích thước của mỗi tham số 'a' phải khớp với `out_features` của lớp
             `PriorFunction` tương ứng trước đó.
           - Khởi tạo bằng 0 và lưu vào `nn.ParameterList`.
        8. Khởi tạo tham số trung vị (`self._medians`):
           - Đây là tham số học được, đại diện cho giá trị trung vị ước lượng
             của phân phối cho mỗi kênh.
           - Khởi tạo bằng 0. Shape: (1, in_channels, 1, 1).
        """
        super(HyperPrior, self).__init__()
        
        # Set device based on availability or user preference
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing HyperPrior on device: {self.device}")

        self.in_channels = in_channels

        # Xây dựng chuỗi kích thước đầy đủ: [1, dim1, dim2, ..., dimN, 1]
        self.transform_dims_full = [1] + list(transform_dims) + [1]
        self.num_chained_layers = len(self.transform_dims_full) - 1

        # Tính scale cơ sở để phân bổ đều init_scale
        # Sử dụng 1.0 để đảm bảo phép chia là float
        base_scale_factor = init_scale**(1.0 / self.num_chained_layers)

        # Tạo các lớp biến đổi affine (PriorFunction)
        prior_functions = []
        for i in range(self.num_chained_layers):
            current_in_features = self.transform_dims_full[i]
            current_out_features = self.transform_dims_full[i+1]

            # Tính scale khởi tạo theo công thức gốc (đảm bảo khởi tạo gần Logistic)
            # np.log(np.expm1(x)) tương đương với np.log(np.exp(x) - 1) nhưng ổn định hơn
            init_prior_scale = np.log(np.expm1(1.0 / base_scale_factor / current_out_features))

            # Tạo lớp PriorFunction
            pf = PriorFunction(
                parallel_dims=self.in_channels,
                in_features=current_in_features,
                out_features=current_out_features,
                scale=init_prior_scale,
                bias=True, # Luôn dùng bias theo bài báo
                device=self.device
            )
            prior_functions.append(pf)
        # Lưu vào ModuleList
        self.affine_transforms = nn.ModuleList(prior_functions)

        # Tạo các tham số 'a' cho phần phi tuyến
        a_params = []
        # Chỉ cần N-1 tham số 'a' cho N lớp affine
        for i in range(self.num_chained_layers - 1):
            # Kích thước của 'a' khớp với output của lớp affine thứ i
            out_dim = self.transform_dims_full[i + 1]
            param = nn.Parameter(torch.zeros(self.in_channels, 1, 1, 1, out_dim)) 
            a_params.append(param)
        # Lưu vào ParameterList
        self.nonlinear_factors = nn.ParameterList(a_params)

        # Tham số trung vị học được
        self._medians = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        
        # Move entire model to the device
        self.to(self.device)

    @property
    def medians(self):
        """
        Thuộc tính (property) để truy cập giá trị trung vị đã học.

        Trả về tensor `_medians` nhưng đã được detach khỏi đồ thị tính toán,
        nghĩa là không tính gradient qua nó khi truy cập bằng cách này.
        Hữu ích khi chỉ cần đọc giá trị median mà không ảnh hưởng đến quá trình backward.
        """
        return self._medians.detach()

    # Replace the problematic cdf method with this implementation:

    def cdf(self, x, logits=True, detach=False):
        """
        Tính toán Hàm Phân Phối Tích Lũy (Cumulative Distribution Function - CDF).
        """
        # Move input to device
        x = x.to(self.device)
        
        # If detach is True, detach the input to prevent gradient flow here
        if detach:
            x = x.detach()
        
        # 1. Chuẩn bị đầu vào
        x_transformed = x.transpose(0, 1)
        x_transformed = x_transformed.unsqueeze(-1)
        
        # Get parameters, detach them if needed
        current_a = [a.detach() if detach else a for a in self.nonlinear_factors]
        
        # 2. Apply transformations
        for i in range(self.num_chained_layers - 1):
            affine_layer = self.affine_transforms[i]
            # Apply PriorFunction layer
            x_transformed = affine_layer(x_transformed, detach=detach)
            
            # Apply nonlinearity SAFELY
            # First, clamp values to prevent extreme values that could lead to NaNs
            safe_a = torch.clamp(current_a[i], -10, 10)
            safe_x = torch.clamp(x_transformed, -10, 10)
            
            # Then compute tanh safely
            a_tanh = torch.tanh(safe_a)
            x_tanh = torch.tanh(safe_x)
            
            # Compute the transformation with additional safety
            x_transformed = x_transformed + a_tanh * x_tanh
        
        # 3. Apply the final affine transformation
        last_affine_layer = self.affine_transforms[-1]
        x_transformed = last_affine_layer(x_transformed, detach=detach)
        
        # 4. Format output
        output = x_transformed.squeeze(-1)
        output = output.transpose(0, 1)
        
        # 5. Return result
        if logits:
            return output  # Return logits
        else:
            return torch.sigmoid(output)  # Return CDF probabilities [0, 1]

    def pdf(self, x):
        """
        Tính toán Hàm Mật Độ Xác Suất (Probability Density Function - PDF).
        PDF(x) là đạo hàm của CDF(x) theo x.

        Sử dụng `torch.autograd.grad` để tính đạo hàm tự động.

        Args:
            x (Tensor): Tensor đầu vào. Shape: (N, C, H, W).

        Returns:
            Tensor: Giá trị PDF đã tính toán. Shape: (N, C, H, W).
        """
        # Move input to device
        x = x.to(self.device)
        
        # Đảm bảo x yêu cầu gradient để có thể tính đạo hàm theo nó
        x = x.requires_grad_(True)

        # Tính CDF (dưới dạng xác suất, không detach)
        cdf_val = self.cdf(x, logits=False, detach=False)

        # Tạo tensor chứa gradient đầu ra (gradient của loss cuối cùng theo cdf_val)
        # Ở đây, chúng ta muốn đạo hàm của chính cdf_val, nên grad_outputs là 1.
        grad_outputs = torch.ones_like(cdf_val)

        # Tính đạo hàm của cdf_val theo x
        # outputs=cdf_val, inputs=x
        # create_graph=True nếu bạn cần tính đạo hàm bậc cao hơn nữa
        pdf_val = torch.autograd.grad(outputs=cdf_val,
                                       inputs=x,
                                       grad_outputs=grad_outputs,
                                       create_graph=True)[0] # Lấy phần tử đầu tiên vì grad trả về tuple

        return pdf_val

    def prior_probability_loss(self):
        """
        Tính toán loss phụ trợ (auxiliary loss) để ổn định quá trình huấn luyện.

        Mục tiêu: "Ép" mô hình học sao cho CDF(median) ≈ 0.5 (tương đương logit ≈ 0).
        Điều này giúp tham số `_medians` học đúng giá trị trung vị thực sự của phân phối.

        Returns:
            Tensor: Một giá trị scalar đại diện cho loss trung vị.
        """
        target_logit = 0.0  # Giá trị logit mục tiêu tại median

        # Tính CDF logits tại giá trị median đã học (_medians).
        # Sử dụng detach=True vì loss này chỉ nhằm cập nhật _medians,
        # không nên ảnh hưởng gradient của các tham số khác qua bước này.
        median_logits = self.cdf(self._medians, logits=True, detach=True)

        # Tính tổng trị tuyệt đối của độ lệch so với target_logit.
        # sum() để ra giá trị loss cuối cùng là scalar.
        extra_loss = torch.sum(torch.abs(median_logits - target_logit))

        return extra_loss

    def likelihood(self, x, min_likelihood=1e-9):
        """
        Tính toán likelihood (khả năng xảy ra) của các giá trị đã được lượng tử hóa `x`.
        Đây là phương thức **quan trọng nhất** cho việc mã hóa entropy.

        Xấp xỉ xác suất P(Biến ngẫu nhiên = x) bằng P(x - 0.5 < Biến ngẫu nhiên <= x + 0.5),
        tức là bằng CDF(x + 0.5) - CDF(x - 0.5).

        Args:
            x (Tensor): Tensor latent đã được lượng tử hóa (thường là số nguyên).
                        Shape: (N, C, H, W).
            min_likelihood (float): Giá trị cận dưới cho likelihood để tránh log(0)
                                    khi tính toán số bit (rate = -log2(likelihood)).

        Returns:
            Tensor: Likelihood đã tính toán cho mỗi giá trị trong x. Shape: (N, C, H, W).
        """
        # Move input to device
        # x = x.to(self.device)
        
        # 1. Tính CDF logits tại các điểm biên của khoảng lượng tử hóa:
        lower_logit = self.cdf(x - 0.5, logits=True, detach=False) # logit(CDF(x - 0.5))
        upper_logit = self.cdf(x + 0.5, logits=True, detach=False) # logit(CDF(x + 0.5))

        # 2. Tính hiệu CDF(upper) - CDF(lower) một cách ổn định về số học:
        #    Sử dụng kỹ thuật trong bài báo gốc để tránh lỗi khi giá trị gần 0 hoặc 1.
        #    sign = -torch.sign(lower + upper)
        #    likelihood ≈ |sigmoid(upper * sign) - sigmoid(lower * sign)|
        sign = -torch.sign(lower_logit + upper_logit)
        # Đảm bảo sign không bao giờ bằng 0 (trường hợp cực hiếm nhưng có thể xảy ra)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        sign = sign.detach() # Không cần gradient qua sign

        upper_prob = torch.sigmoid(upper_logit * sign)
        lower_prob = torch.sigmoid(lower_logit * sign)

        likelihood_val = torch.abs(upper_prob - lower_prob)

        # 3. Áp dụng cận dưới bằng hàm LowerBound tùy chỉnh:
        #    Đảm bảo likelihood >= min_likelihood mà vẫn giữ gradient đúng.
        likelihood_val = LowerBound.apply(likelihood_val, min_likelihood)

        return likelihood_val

    def inverse_cdf(self, xi, method='bisection', max_iterations=100, tol=1e-6, **kwargs):
        """
        Tính toán Hàm Phân Phối Tích Lũy Ngược (Inverse CDF hay Quantile Function).
        Tìm giá trị 'z' sao cho CDF(z) = xi, với 'xi' là một xác suất (thường từ 0 đến 1).

        Sử dụng phương pháp tìm kiếm nhị phân (bisection) để giải phương trình CDF(z) - xi = 0.

        Args:
            xi (Tensor): Các giá trị xác suất đầu vào. Shape: (N, C, H, W).
            method (str): Phương pháp sử dụng ('bisection').
            max_iterations (int): Số vòng lặp tối đa cho tìm kiếm nhị phân.
            tol (float): Ngưỡng dung sai để dừng tìm kiếm (khi khoảng cách giữa
                         hai đầu mút đủ nhỏ).

        Returns:
            Tensor: Các giá trị 'z' tương ứng với xác suất 'xi'. Shape: (N, C, H, W).
        """
        # Move input to device
        xi = xi.to(self.device)
        
        if method == 'bisection':
            # --- Hàm mục tiêu f(z) = CDF(z) - xi ---
            # Ta cần tìm z sao cho f(z) = 0.
            # Sử dụng detach=True vì icdf thường dùng cho sampling, không cần gradient.
            def f(z_):
                return self.cdf(z_, logits=False, detach=True) - xi

            # --- Khởi tạo khoảng tìm kiếm [left, right] ---
            # Cần đảm bảo f(left) < 0 và f(right) > 0.
            # Bắt đầu với một khoảng lớn và mở rộng nếu cần.
            # Sử dụng giá trị lớn hơn [-1, 1] để an toàn hơn.
            init_interval = torch.tensor([-100.0, 100.0], device=self.device)
            left_endpoints = torch.full_like(xi, init_interval[0])
            right_endpoints = torch.full_like(xi, init_interval[1])

            # Mở rộng khoảng về bên trái (âm) nếu f(left) >= 0
            iter_count = 0
            max_expand_iters = 20 # Giới hạn số lần mở rộng
            while torch.any(f(left_endpoints) >= 0) and iter_count < max_expand_iters:
                left_endpoints = left_endpoints * 1.5 # Mở rộng theo tỉ lệ
                iter_count += 1
            # Mở rộng khoảng về bên phải (dương) nếu f(right) <= 0
            iter_count = 0
            while torch.any(f(right_endpoints) <= 0) and iter_count < max_expand_iters:
                right_endpoints = right_endpoints * 1.5
                iter_count += 1
            if iter_count == max_expand_iters:
                 print("Warning: ICDF interval expansion reached max iterations.")


            # --- Thực hiện tìm kiếm nhị phân ---
            for i in range(max_iterations):
                # Tính điểm giữa
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                # Tính giá trị hàm mục tiêu tại điểm giữa
                mid_vals = f(mid_pts)

                # Cập nhật đầu mút dựa trên dấu của mid_vals
                # Nếu mid_vals > 0, nghiệm nằm ở [left, mid] -> cập nhật right = mid
                # Nếu mid_vals < 0, nghiệm nằm ở [mid, right] -> cập nhật left = mid
                # Nếu mid_vals = 0, đã tìm thấy nghiệm (nhưng vẫn cập nhật để thu hẹp)
                is_pos = mid_vals > 0
                is_neg = mid_vals < 0
                right_endpoints = torch.where(is_pos, mid_pts, right_endpoints)
                left_endpoints = torch.where(is_neg, mid_pts, left_endpoints)

                # Kiểm tra điều kiện dừng: khoảng cách giữa hai đầu mút đủ nhỏ
                interval_width = right_endpoints - left_endpoints
                if torch.all(interval_width <= tol):
                    # print(f'bisection terminated after {i+1} its')
                    break # Thoát vòng lặp sớm
            else: # Thực thi nếu vòng lặp kết thúc mà không break
                print(f'Warning: ICDF bisection reached max iterations ({max_iterations}). Final interval width: {torch.max(interval_width)}')

            # Trả về điểm giữa của khoảng cuối cùng làm kết quả xấp xỉ
            return 0.5 * (left_endpoints + right_endpoints)
        else:
            raise NotImplementedError(f"ICDF method '{method}' not implemented.")

    def sample(self, shape, device=None):
        """
        Tạo mẫu ngẫu nhiên từ phân phối xác suất đã học được bởi HyperPrior.

        Sử dụng phương pháp Inverse Transform Sampling:
        1. Tạo các số ngẫu nhiên `u` từ phân phối đều trong khoảng (0, 1).
        2. Tính `z = ICDF(u)`. Các giá trị `z` này sẽ tuân theo phân phối đã học.

        Args:
            shape (tuple): Kích thước (shape) mong muốn của tensor mẫu đầu ra
                           (ví dụ: (N, C, H, W)).
            device: Thiết bị (cpu hoặc cuda) để tạo tensor mẫu trên đó.
                    Nếu None, sẽ sử dụng thiết bị của mô hình.

        Returns:
            Tensor: Các mẫu ngẫu nhiên được tạo ra. Shape giống như `shape` đầu vào.
        """
        # Use model's device if none provided
        if device is None:
            device = self.device
            
        # Tạo số ngẫu nhiên đều trong khoảng (eps, 1-eps) để tránh các giá trị
        # cực biên 0 và 1 có thể gây lỗi số học trong icdf.
        eps = 1e-6
        uniform_random = torch.rand(shape, device=device) * (1.0 - 2 * eps) + eps

        # Áp dụng icdf để biến đổi các số ngẫu nhiên đều thành mẫu từ phân phối prior
        samples = self.inverse_cdf(uniform_random)
        return samples
    


# # Testing HyperPrior class
# if __name__ == "__main__":
#     # Khởi tạo HyperPrior với các tham số mẫu
#     hyper_prior = HyperPrior(in_channels=256, transform_dims=[3, 3, 3], init_scale=10.0)

#     # Tạo một tensor ngẫu nhiên để thử nghiệm
#     x = torch.randn(32, 256, 32, 32)  # Ví dụ: batch_size=1, channels=256, height=32, width=32

#     # Tính toán CDF và PDF
#     cdf_val = hyper_prior.cdf(x)
#     pdf_val = hyper_prior.pdf(x)

#     # In kết quả
#     print("CDF Value:", cdf_val.shape)
#     print("PDF Value:", pdf_val.shape)
#     # Tính toán likelihood
#     likelihood_val = hyper_prior.likelihood(x)
#     print("Likelihood Value:", likelihood_val)
#     # Tính toán inverse CDF
#     xi = torch.rand(1, 256, 32, 32)  # Ví dụ: batch_size=1, channels=256, height=32, width=32
#     inverse_cdf_val = hyper_prior.inverse_cdf(xi)
#     print("Inverse CDF Value:", inverse_cdf_val.shape)
#     # Tính toán loss phụ trợ
#     prior_loss = hyper_prior.prior_probability_loss()
#     print("Prior Probability Loss:", prior_loss.item())
#     # Tạo mẫu ngẫu nhiên
#     samples = hyper_prior.sample((1, 256, 32, 32), device='cpu')
#     print("Random Samples:", samples.shape)
#     # Kiểm tra medians
#     medians = hyper_prior.medians
#     print("Medians Value:", medians.shape)
#     # Kiểm tra likelihood với cận dưới