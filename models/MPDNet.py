import torch
import torch.nn as nn
from layers.p_networks import RevIN, STLDecompose, FITS, Freq_TVT, TrendPredict, SeasonalPredict, LocalPredict
from statsmodels.tsa.filters.hp_filter import hpfilter

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.period = configs.period[:configs.k]
        self.revin = RevIN(configs.enc_in)
        self.decomp = STLDecompose(self.period)
        self.trendPredict = TrendPredict(configs.seq_len, configs.pred_len, configs.patch_len, configs.d_model,
                                         configs.k, configs.dropout)
        self.seasonalPredict = SeasonalPredict(configs.seq_len, configs.pred_len, 32,
                                         configs.k, configs.dropout, configs.enc_in, self.period)

        self.localPredict = LocalPredict(configs.k, self.period, configs.seq_len, configs.pred_len, configs.d_model,
                                         configs.kernel_list)
        # self.localPredict = nn.Linear(configs.seq_len, configs.pred_len)
        # self.seasonalPredict = Freq_TVT(configs)


    def forward(self, x, x_mark, y_true, y_mark):
        x = self.revin(x, 'norm')
        # seq_last = x[:, -1:, :].detach()
        # x = x - seq_last
        B, L, D = x.shape
        trends, seasonals = self.decomp(x)
        trends = torch.stack(trends, dim=1)
        seasonals = torch.stack(seasonals, dim=1)
        trends = trends.permute(0, 3, 1, 2).reshape(B * D, -1, L)
        seasonals = seasonals.permute(0, 3, 1, 2)#.reshape(B * D, -1, L)
        trend_output = self.trendPredict(trends)
        seasonal_output = self.seasonalPredict(seasonals, y_mark)
        local_output = self.localPredict(seasonals)
        output = trend_output + seasonal_output #+ local_output
        # output = seasonal_output
        # output = local_output
        output = output.reshape(B, D, -1).transpose(1, 2)
        # output = output + seq_last
        output = self.revin(output, 'denorm')
        return output







