import torch
import torch.nn as nn
import math
from hgru import HgruRealV6, Hgru1dV3, BiHgru1d
import torch.nn.functional as F
import functools

def cal_padding(input_len, seg_len):
    padding_len = seg_len - (input_len % seg_len) if input_len % seg_len != 0 else 0
    return padding_len


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

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

class TrendPredict(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, d_model, k, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.trend_init = nn.Linear(seq_len, pred_len)
        self.trend_padding = cal_padding(seq_len + pred_len, patch_len)
        # self.trend_padding = cal_padding(pred_len, patch_len)
        self.linear_patch = nn.Conv1d(in_channels=k, out_channels=d_model, kernel_size=patch_len, stride=patch_len)
        self.linear_patch_re = nn.ConvTranspose1d(in_channels=d_model, out_channels=1, kernel_size=patch_len,
                                                       stride=patch_len)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.d_model = d_model
        self.hgru = HgruRealV6(d_model)
        # self.hawk = Hawk(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # self.pos = nn.Parameter(torch.randn(1, k, (self.trend_padding + seq_len + pred_len)))

    def forward(self, x):
        B, C, L = x.shape
        trend_pred = self.trend_init(x)
        trend = torch.cat([x, trend_pred], dim=-1)
        # trend = trend_pred
        last = trend[:, :, -1].unsqueeze(-1).repeat(1, 1, self.trend_padding)
        trend = torch.cat([trend, last], dim=-1)
        # trend = trend + self.pos
        # trend = self.norm(trend)
        # trend = trend.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # trend = trend.reshape(B * C, -1, self.patch_len)
        trend_in = self.linear_patch(trend)
        trend_in = trend_in.transpose(1, 2)
        trend_in = torch.cat([self.time_shift(trend_in[:, :, :self.d_model // 2]), trend_in[:, :, self.d_model // 2:]],
                             dim=-1)
        trend_in = self.act(trend_in)
        trend_out = self.hgru(trend_in)
        # trend_out = self.hawk(trend_in)
        # trend_out = trend_in
        # trend_out = self.norm(trend_in + self.dropout(trend_out)).transpose(1, 2)
        # trend_out = self.linear_patch_re(trend_out.transpose(1, 2))
        trend_out = self.linear_patch_re(self.dropout(trend_out.transpose(1, 2)))
        trend_out = trend_out[:, :, :(self.seq_len + self.pred_len)]
        # trend_out = trend_out[:, :, :self.pred_len]
        trend_output = trend_out[:, :, -self.pred_len:]
        return trend_output

class SeasonalPredict(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, k, dropout, enc_in, periods):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.len = (seq_len + pred_len)
        # self.embedding = nn.Conv1d(in_channels=k, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.fno = nn.Parameter(0.02 * torch.randn(self.seq_len // 2 + 1, self.pred_len // 2 + 1, dtype=torch.cfloat))
        # self.fno = nn.Parameter(0.02 * torch.randn(self.len // 2 + 1, self.len // 2 + 1, dtype=torch.cfloat))
        # self.pred_proj = nn.Linear(self.seq_len, self.pred_len)
        self.out_proj = nn.Conv1d(in_channels=k, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Linear(4, enc_in)
        self.revIn= RevIN(k*enc_in)
        # self.periodEmbedd = self._getPeriodEmbedd(periods)#nn.Parameter(torch.randn(1, 1, k, seq_len + pred_len))
        # self.norm= nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        # self.norm = nn.LayerNorm(d_model)
        # self.length_ratio = (seq_len + pred_len) // seq_len
        # self.dominance_freq = seq_len // 2
        # self.fno = nn.Parameter(0.02 * torch.randn(self.dominance_freq,
        #                                            self.dominance_freq * self.length_ratio, dtype=torch.cfloat))


    def forward(self, x, x_mark=None):
        # x = self.embedding(x)
        # x = self.dropout(self.act(x))
        # x = self.act(x)
        B, D, K, L = x.shape
        # x = x.reshape(B, -1, L).transpose(1, 2)
        # x = self.revIn(x, 'norm').transpose(1, 2)

        # zeros = torch.zeros([B, D, K, self.pred_len], device=x.device)
        # x = torch.cat([x, zeros], dim=-1)
        # pos = self.embed(x_mark).transpose(1, 2)
        # x = x + pos.unsqueeze(dim=2) #+ self.periodEmbedd.to(x.device)
        x = x.reshape(B*D, K, -1)
        x_fft = torch.fft.rfft(x, dim=-1)
        out_fft = torch.einsum("bcx,xl->bcl", x_fft, self.fno)
        output = torch.fft.irfft(out_fft, dim=-1)
        # output = output.reshape(B, D*K, -1).transpose(1, 2)
        # output = self.revIn(output, 'denorm')
        # output = output.transpose(1, 2).reshape(B*D, K, -1)
        # output = self.act(output)
        # output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        # output = self.pred_proj(x)
        out = self.out_proj(output)

        return out[:, :, -self.pred_len:]

class LoaclExtract(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, p, kernel_list):
        super().__init__()
        self.inPadding = cal_padding(seq_len, p)
        self.in_nums = (seq_len + p - 1) // p
        self.out_nums = (pred_len + p - 1) // p
        self.period = p
        self.pred_len = pred_len
        self.conv1 = nn.Conv1d(self.in_nums, d_model, kernel_size=3, padding=3 // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(d_model, self.out_nums, kernel_size=3, padding=3 // 2)


    def forward(self, x):
        seasonal_padding = torch.zeros([x.shape[0], self.inPadding]).to(x.device)
        seasonal = torch.cat([x, seasonal_padding], dim=-1)
        seasonal = seasonal.unfold(dimension=1, size=self.period, step=self.period)
        seasonal_out = self.conv1(seasonal)
        seasonal_out = self.act(seasonal_out)
        seasonal_out = self.conv2(seasonal_out)
        seasonal_out = seasonal_out.transpose(1, 2).reshape(-1, self.out_nums * self.period)[:, :self.pred_len]
        return seasonal_out



class LocalPredict(nn.Module):
    def __init__(self, k, periods, seq_len, pred_len, d_model, kernel_list):
        super().__init__()
        self.k = k
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.backbone = nn.ModuleList([LoaclExtract(seq_len, pred_len, d_model, periods[i], kernel_list)
                                       for i in range(self.k)])
        self.out_proj = nn.Conv1d(in_channels=k, out_channels=1, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        B, D, K, L = x.shape
        x = x.reshape(B*D, K, L)
        local = []
        for i in range(self.k):
            out = self.backbone[i](x[:, i, :])
            local.append(out)
        local = torch.stack(local, dim=1)
        return self.out_proj(local)


# from accelerated_scan.warp import scan
from torch.nn.functional import softplus, gelu
from hgru.hgru_real_cuda import HgruRealFunction
triton_parallel_scan = HgruRealFunction.apply

class Hawk(nn.Module):
    def __init__(self, dim, act_fun="gelu", causal=True, use_triton=True, kernel_size=4, expansion_factor=1.5):
        super().__init__()
        # params = locals()
        # print_params(**params)
        # embed_dim = int(dim * expansion_factor)
        embed_dim = dim
        self.input = nn.Linear(dim, 2 * embed_dim, bias=False)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, bias=True,
                              kernel_size=kernel_size, groups=embed_dim, padding=kernel_size - 1)
        self.gates = nn.Linear(embed_dim, 2*embed_dim, bias=True)
        self.forget_base = nn.Parameter(torch.linspace(-4.323, -9, embed_dim))
        self.output = nn.Linear(embed_dim, dim, bias=False)
        self.alpha_log_scale = nn.Parameter(torch.tensor([8]).log(), requires_grad=False)
        self.scan = HgruRealFunction.apply if not use_triton else triton_parallel_scan
        with torch.no_grad():
            self.input.weight.normal_(std=dim ** -0.5)
            self.gates.weight.normal_(std=embed_dim ** -0.5)
            self.output.weight.normal_(std=embed_dim ** -0.5)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _N, T, _C = x.shape
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT

        # RG-LRU: linear recurrent unit with input-dependent gating
        forget, input = self.gates(x).chunk(2, dim=-1)
        alpha = (-self.alpha_log_scale.exp() * softplus(self.forget_base) * forget.sigmoid()).exp()
        beta = (1 - alpha ** 2 + 1e-6).sqrt()  # stabilizes variance
        x = beta * input.sigmoid() * x
        h = self.scan(x, alpha)
        # h = self.norm(h)
        # h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
        x = self.output(gelu(gate) * h)

        return x

class GatedMLP(nn.Module):
    def __init__(self, dim=1024, expansion_factor=2):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = nn.Linear(dim, 2 * hidden, bias=False)
        self.shrink = nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.grow.weight.normal_(std=dim**-0.5)
            self.shrink.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = gelu(gate) * x
        return self.shrink(x)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return self.gamma / self.scale * x

class Griffin(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.hawk_norm = RMSNorm(dim=d_model)
        self.hawk = Hawk(dim=d_model)
        self.hawk_gmlp_norm = RMSNorm(dim=d_model)
        self.hawk_gmlp = GatedMLP(dim=d_model)

    def forward(self, x):
        x = x + self.hawk(self.hawk_norm(x))
        x = x + self.hawk_gmlp(self.hawk_gmlp_norm(x))
        return x





def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# class seasonality_extraction(nn.Module):
#     def __init__(self):
#         super

class decompose(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.moving_avg = [moving_avg(p, stride=1) for p in period]

    def forward(self, x):
        trend = []
        seasonal = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            trend.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

class STLDecompose(nn.Module):
    def __init__(self, periods, seq_len=None):
        super().__init__()
        self.periods = periods
        # self.seq_len = seq_len
        self.moving_avg = [moving_avg(period, stride=1) for period in periods]
        # self.seasonal_extract = [seasonality_extraction(period, (seq_len + period - 1) // period) for period in periods]

    def forward(self, x):
        trends = []
        seasonals = []
        for i in range(len(self.periods)):
            moving_avg = self.moving_avg[i](x)
            trends.append(moving_avg)
            seasonal = x - moving_avg
            seasonals.append(seasonal)
        return trends, seasonals

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        # self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=filter_size, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=filter_size, out_channels=hidden_size, kernel_size=1, bias=False)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)
        # self.initialize_conv(self.conv1)
        # self.initialize_conv(self.conv2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)

        # x = self.conv1(x.transpose(1, 2))
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.conv2(x).transpose(1, 2)
        return x

    def initialize_conv(self, x):
        if isinstance(x, nn.Conv1d):
            # nn.init.kaiming_normal_(x.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))
    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

class Freq_TVT2(nn.Module):
    def __init__(self, configs):
        super(Freq_TVT2, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[configs.freq]
        self.embed = nn.Linear(d_inp, configs.enc_in)
        self.pred_len = configs.pred_len
        self.output_len = configs.pred_len
        self.len = self.output_len //2 + 1
        self.fno = nn.Parameter(0.02 * torch.randn(configs.seq_len // 2 + 1, self.len, dtype=torch.cfloat))
        self.bias = nn.Parameter(0.02 * torch.randn(configs.enc_in, self.len, dtype=torch.cfloat))
        self.act = nn.GELU()
        self.ffn = FeedForwardNetwork(self.output_len, configs.d_ff, configs.dropout)
        self.drop = nn.Dropout(configs.dropout)

        self.out = nn.Linear(configs.seq_len, configs.pred_len)


    def forward(self, input, x_mark=None):
        x = input.transpose(1, 2)
        x_fft = torch.fft.rfft(x, dim=-1)
        out_fft = torch.einsum("bcx,xl->bcl", x_fft, self.fno) #+ self.bias
        x_fft = out_fft
        output = torch.fft.irfft(x_fft, dim=-1, n=self.output_len)
        # output = self.drop(self.act(self.ffn(output)))
        return output.transpose(1, 2)

class Freq_TVT(nn.Module):
    def __init__(self,  configs):
        super(Freq_TVT, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[configs.freq]
        in_channels = configs.enc_in
        d_ff = configs.d_ff
        self.seq_len = seq_len = configs.seq_len
        self.pred_len = pred_len = configs.pred_len
        dropout = configs.dropout
        self.embed = nn.Linear(d_inp, configs.k)
        self.pred_len = pred_len
        self.len = (seq_len + pred_len) //2 + 1
        # self.len = seq_len // 2 + 1
        self.output_len = seq_len + pred_len
        self.fno = nn.Parameter(0.02 * torch.randn(self.len, self.len, dtype=torch.cfloat))
        self.bias = nn.Parameter(0.02 * torch.randn(self.len, dtype=torch.cfloat))
        self.act = nn.GELU()
        self.ffn = FeedForwardNetwork(pred_len, d_ff, dropout)
        self.drop = nn.Dropout(dropout)
        # self.channel_attention = MultiheadAttention(self.len, n_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_ff)


    def forward(self, input, x_mark=None):
        zeros = torch.zeros([input.shape[0], self.pred_len, input.shape[2]], device=input.device)
        input = torch.cat([input[:, -self.seq_len:, :], zeros], dim=1)
        B, L, D = input.shape
        input = input.transpose(1, 2)
        pos = self.embed(x_mark).transpose(1, 2)
        x = input + pos
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_fft = torch.einsum("bcx,xl->bcl", x_fft, self.fno) #+ self.bias
        x_fft = out_fft
        output = torch.fft.irfft(x_fft, dim=-1, norm='ortho')
        # output = self.drop(self.act(self.ffn(output)))
        output = output[:, :, -self.pred_len:]
        return output.transpose(1, 2)

class TSMamba(nn.Module):
    def __init__(self, configs, decomp, period):
        super(TSMamba, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.decomp = decomp
        # self.mamba = Block(dim=configs.d_model,
        #                    mixer_cls=Mamba(d_model=configs.d_model, d_state=16,d_conv=4,expand=2),
        #                    norm_cls=nn.LayerNorm(configs.d_model))
        # self.hgru = HgruRealV6(self.d_model, self.pred_len // self.patch_len)
        self.lru = LRU(configs.d_model, self.pred_len // self.patch_len)
        # self.rev_lru = LRU(configs.d_model, self.pred_len // self.patch_len)
        self.freq = Freq_TVT(configs)
        # self.freqSelect = FrequenceSelect(self.seq_len, self.enc_in)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(configs.dropout)
        self.linear_patch = nn.Linear(configs.patch_len, configs.d_model)
        # self.linear_patch_re = nn.Linear(self.seq_len // self.patch_len * self.d_model, self.pred_len)
        self.linear_patch_re = nn.Linear(configs.d_model, self.patch_len)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        N = self.seq_len // self.patch_len
        B, L, C = trend.shape
        trend = trend.reshape(B, -1, N, C).permute(0, 3, 2, 1).reshape(B*C, N, -1)
        trend_in = self.linear_patch(trend)
        # trend_out = self.mamba(trend_in)

        trend_in = self.time_shift(trend_in)
        trend_out = self.lru(trend_in)
        # trend_out = self.rev_lru(torch.flip(trend_out, [1]))
        # trend_out = torch.flip(trend_out, dims=[1])
        # trend_out = self.hgru(trend_in)
        # trend_out = trend_out.transpose(1, 2)
        # trend_out = self.linear_patch_re(self.dropout(trend_out.flatten(1)))

        trend_out = self.linear_patch_re(self.dropout(trend_out))
        # trend_out = trend_out.reshape(B, C, -1).permute(0, 2, 1)
        trend_out = trend_out.transpose(1, 2).reshape(B, C, -1).permute(0, 2, 1)
        trend_out = self.decomp(trend_out)[1]
        # seasonal = self.freqSelect(seasonal)

        seasonal_out = self.freq(seasonal)
        out = trend_out + seasonal_out
        # out = seasonal_out
        return out

class FrequenceSelect(nn.Module):
    def __init__(self, seq_len, pred_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.input_len = seq_len // 2 + 1
        self.d_model = d_model
        self.scale = 0.02
        self.sparsity_threshold = 0.01
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.d_model, self.input_len, self.input_len))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.d_model, self.input_len))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.d_model, self.input_len, self.input_len))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.d_model, self.input_len))
        self.act = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        fft_x = torch.fft.rfft(x, dim=-1)
        o1_real = (
            torch.einsum('bcl,clo->bco', fft_x.real, self.w1[0]) - \
            torch.einsum('bcl,clo->bco', fft_x.imag, self.w1[1]) + \
            self.b1[0, :, :]
        )

        o1_imag = (
            torch.einsum('bcl,clo->bco', fft_x.imag, self.w1[0]) + \
            torch.einsum('bcl,clo->bco', fft_x.real, self.w1[1]) + \
            self.b1[1, :, :]
        )

        # o2_real = (
        #         torch.einsum('bcl,clo->bco', o1_real, self.w2[0]) - \
        #         torch.einsum('bcl,clo->bco', o1_imag, self.w2[1]) + \
        #         self.b2[0, :, :]
        # )
        #
        # o2_imag = (
        #         torch.einsum('bcl,clo->bco', o1_imag, self.w2[0]) + \
        #         torch.einsum('bcl,clo->bco', o1_real, self.w2[1]) + \
        #         self.b2[1, :, :]
        # )
        gate = torch.stack([o1_real, o1_imag], dim=-1)
        gate = F.softshrink(gate, lambd=self.sparsity_threshold)
        gate = torch.view_as_complex(gate)
        fft_x = fft_x * gate
        out = torch.fft.irfft(fft_x)
        return out.transpose(1, 2)


class FITS(nn.Module):
    # FITS: Frequency Interpolation Time Series Forecasting
    def __init__(self, configs):
        super(FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.dominance_freq = 30
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
            torch.cfloat)  # complex layer for frequency upcampling]
        self.out_proj = nn.Conv1d(in_channels=configs.k, out_channels=1, kernel_size=3, stride=1, padding=1)
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len
        # self.embedding = nn.Conv1d(in_channels=configs.k, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.out_proj = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.act = nn.GELU()
        # self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        # RIN
        # x_mean = torch.mean(x, dim=-1, keepdim=True)
        # x = x - x_mean
        # x_var = torch.var(x, dim=-1, keepdim=True) + 1e-5
        # # print(x_var)
        # x = x / torch.sqrt(x_var)
        # x = self.dropout(self.act(self.embedding(x)))
        low_specx = torch.fft.rfft(x, dim=-1)
        low_specx[:, :, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, :, 0:self.dominance_freq]  # LPF
        # print(low_specx.permute(0,2,1))
        low_specxy_ = self.freq_upsampler(low_specx)
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), int((self.seq_len + self.pred_len) / 2 + 1)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, :, 0:low_specxy_.size(2)] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=-1)
        low_xy = low_xy * self.length_ratio  # compemsate the length change
        # dom_x=x-low_x

        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        # xy = (low_xy) * torch.sqrt(x_var) + x_mean
        # xy = self.out_proj(low_xy)
        xy = self.out_proj(low_xy)
        return xy[:, :, -self.pred_len:]#, low_xy * torch.sqrt(x_var)