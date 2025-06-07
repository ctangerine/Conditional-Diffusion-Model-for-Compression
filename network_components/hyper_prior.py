import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network_components.utils import LowerBound

class PriorFunction(nn.Module):
    #  A Custom Function described in Balle et al 2018. https://arxiv.org/pdf/1802.01436.pdf
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True):
        super(PriorFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, in_features, out_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, 1, out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(scale)

        

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, input, detach=False):
        # input shape (channel, batch_size, in_features)
        if detach:
            return torch.matmul(input, F.softplus(self.weight.detach())) + self.bias.detach()
        return torch.matmul(input, F.softplus(self.weight)) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)


class HyperPrior(nn.Module):
    '''
        A prior model described in Balle et al 2018 Appendix 6.1 https://arxiv.org/pdf/1802.01436.pdf
        return the boxshape likelihood
    '''
    def __init__(self, channels=192, dims=[3, 3, 3], init_scale=10.):
        super(HyperPrior, self).__init__()
        dims = [1] + dims + [1]
        self.chain_len = len(dims) - 1
        scale = init_scale**(1 / self.chain_len)
        h_b = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.chain_len):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            h_b.append(PriorFunction(channels, dims[i], dims[i + 1], init).to(device))
        self.affine = nn.ModuleList(h_b)
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels, 1, 1, 1, dims[i + 1], device=device)) for i in range(self.chain_len - 1)]
        )

        # optimize the medians to fix the offset issue
        self._medians = nn.Parameter(torch.zeros(1, channels, 1, 1, device=device))
        # self.register_buffer('_medians', torch.zeros(1, channels, 1, 1))

    @property
    def medians(self):
        return self._medians.detach()

    def cdf(self, x, logits=True, detach=False):
        x = x.transpose(0, 1).unsqueeze(-1)  # C, N, H, W, 1
        if detach:
            for i in range(self.chain_len - 1):
                x = self.affine[i](x, detach)
                x = x + torch.tanh(self.a[i].detach()) * torch.tanh(x)
            if logits:
                return self.affine[-1](x, detach).squeeze(-1).transpose(0, 1)
            return torch.sigmoid(self.affine[-1](x, detach)).squeeze(-1).transpose(0, 1)

        # not detached
        for i in range(self.chain_len - 1):
            x = self.affine[i](x)
            x = x + torch.tanh(self.a[i]) * torch.tanh(x)
        if logits:
            return self.affine[-1](x).squeeze(-1).transpose(0, 1)
        return torch.sigmoid(self.affine[-1](x)).squeeze(-1).transpose(0, 1)

    def pdf(self, x):
        cdf = self.cdf(x, False)
        jac = torch.ones_like(cdf)
        pdf = torch.autograd.grad(cdf, x, grad_outputs=jac)[0]
        return pdf

    def get_extraloss(self):
        target = 0
        logits = self.cdf(self._medians, detach=True)
        extra_loss = torch.abs(logits - target).sum()
        return extra_loss

    def likelihood(self, x, min=1e-9):
        lower = self.cdf(x - 0.5, True)
        upper = self.cdf(x + 0.5, True)
        sign = -torch.sign(lower + upper).detach()
        upper = torch.sigmoid(upper * sign)
        lower = torch.sigmoid(lower * sign)
        return LowerBound.apply(torch.abs(upper - lower), min)

    def icdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9, **kwargs):
        if method == 'bisection':
            init_interval = [-1, 1]
            left_endpoints = torch.ones_like(xi) * init_interval[0]
            right_endpoints = torch.ones_like(xi) * init_interval[1]

            def f(z):
                return self.cdf(z, logits=False, detach=True) - xi

            while True:
                if (f(left_endpoints) < 0).all():
                    break
                else:
                    left_endpoints = left_endpoints * 2
            while True:
                if (f(right_endpoints) > 0).all():
                    break
                else:
                    right_endpoints = right_endpoints * 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = torch.logical_not(pos)
                neg = mid_vals < 0
                non_neg = torch.logical_not(neg)
                left_endpoints = left_endpoints * non_neg.float() + mid_pts * neg.float()
                right_endpoints = right_endpoints * non_pos.float() + mid_pts * pos.float()
                if (torch.logical_and(non_pos, non_neg)).all() or torch.min(right_endpoints - left_endpoints) <= tol:
                    print(f'bisection terminated after {i} its')
                    break

            return mid_pts
        else:
            raise NotImplementedError

    def sample(self, img, shape):
        uni = torch.rand(shape, device=img.device)
        return self.icdf(uni)
