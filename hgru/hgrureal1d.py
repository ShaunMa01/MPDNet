import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn


from .helpers import get_activation_fn, print_params

from .hgru_real_cuda import HgruRealFunction

triton_parallel_scan = HgruRealFunction.apply

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return self.gamma / self.scale * x

class HgruRealV6(nn.Module):
    def __init__(
        self,
        embed_dim,
        pred_len=None,
        act_fun="gelu",
        causal=True,
        use_triton=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.lambda_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # self.norm = RMSNorm(embed_dim)
        # self.norm2 = RMSNorm(embed_dim)
        # self.norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2))
        self.act = get_activation_fn(act_fun)
        self.causal = causal
        self.scan = HgruRealFunction.apply if not use_triton else triton_parallel_scan
        self.pred_len = pred_len

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))
        input = (1 - lambda_) * input

        if self.causal:
            hiddens = self.scan(input, lambda_)
        else:
            hiddens_forward = self.scan(input, lambda_)
            hiddens_backward = self.reverse_scan(input, lambda_)
            hiddens = hiddens_forward + hiddens_backward
        h = hiddens[:, -1, :]
        hiddens = self.norm(hiddens)
        output = self.out_proj(hiddens * gate)
        # output = self.out_proj(hiddens)

        output = x + self.norm2(output)

        # output = self.out_proj(hiddens)
        # input_t = output[:, -1, :]
        # out =  torch.zeros(input.shape[0], self.pred_len, input.shape[2]).to(input)
        # out[:, 0, :] = output[:, -1, :]
        # for i in range(1, self.pred_len):
        #     input_t, h = self.foward_step(input_t, h)
        #     out[:, i, :] = input_t
        # return out
        return output#, hiddens

    def reverse_scan(self, input, lambda_):
        hiddens_reverse = self.scan(
            torch.flip(input, dims=[0]),
            torch.flip(lambda_, dims=[0]),
        )

        return torch.flip(hiddens_reverse, dims=[0])

    def forward_naive(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))
        input = (1 - lambda_) * input

        hidden = torch.zeros(1, b, d).to(x)
        hiddens = []
        for i in range(n):
            hidden = lambda_[i] * hidden + input[i]
            hiddens.append(hidden)
        hiddens = torch.cat(hiddens, dim=0)

        hiddens = self.norm(hiddens)
        output = self.out_proj(hiddens * gate)

        return output

    def foward_step(self, x, hidden, lower_bound=0):
        b, d = x.shape
        input_state = self.act(self.input_proj(x))
        gate = self.act(self.gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.lambda_proj(x))
        hidden = lambda_ * hidden + (1 - lambda_) * input_state
        h = hidden
        hidden = self.norm(hidden)
        output = self.out_proj(hidden * gate)
        return output, h
