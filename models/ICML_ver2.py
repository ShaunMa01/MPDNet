import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.Autoformer_EncDec import series_decomp
from layers.Revin import RevIN

class Pre_Block(nn.Module):
    def __init__(self, configs, downsam_time):
        super(Pre_Block, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel_size = configs.enc_in
        self.downsample = downsam_time

        self.l_size = int(self.seq_len*2**(-self.downsample))
        self.revin_layer_1 = RevIN(configs.enc_in)  # norm for input
        # self.revin_layer_2 = RevIN(configs.enc_in)

        # down sampling
        self.conv = nn.Conv1d(in_channels=self.channel_size, out_channels=self.channel_size,
                              kernel_size=2**self.downsample, stride=2**self.downsample)
        self.conv_low = nn.ConvTranspose1d(in_channels=self.channel_size, out_channels=self.channel_size,
                              kernel_size=2, stride=2)
        # mlp
        self.model_type = 'mlp'
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.mapping = nn.Linear(self.l_size, self.l_size)
        elif self.model_type == 'mlp':
            self.mapping = nn.Sequential(
                nn.Linear(self.l_size, self.l_size),
                nn.ReLU(),
                nn.Linear(self.l_size, self.l_size)
            )

        # codebook
        self.codebook = nn.Parameter(torch.randn(self.l_size, self.channel_size))
        # self.codebook = nn.Embedding(self.l_size, self.channel_size)
        # 缺初始化

    def forward(self, x, x_low=None): # [B,T,C]
        if x_low is not None:
            x = x + self.conv_low(x_low)
        x_de = self.conv(x)
        if self.use_revin:
            x_de = self.revin_layer_1(x_de, 'norm')

        distances = torch.cdist(x_de, self.codebook)
        encoding_indices = torch.argmin(distances, dim=-1)
        quantized = F.embedding(encoding_indices, self.codebook)

        x_de_trend = x_de - quantized
        x_de_trend = self.mapping(x_de_trend.permute(0, 2, 1)).permute(0, 2, 1)

        x_de_out = x_de_trend + quantized
        if self.use_revin:
            x_de_out = self.revin_layer_1(x_de_out, 'denorm')
        return x_de_out


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        self.device = torch.device('cuda:0')
        # kernel_size = configs.moving_avg
        # self.decomp = series_decomp(kernel_size)

        # self.revin_layer_1 = RevIN(configs.enc_in)
        # self.revin_layer_2 = RevIN(configs.enc_in)
        # self.revin_layer_3 = RevIN(configs.enc_in)

        # Embedding
        # self.enc_embedding_1 = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)
        # self.enc_embedding_2 = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)
        # self.enc_embedding_3 = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)
        # self.enc_embedding_4 = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)


        self.model = nn.ModuleList([Pre_Block(configs,configs.e_layers-i)
                                    for i in range(configs.e_layers)])

        self.merge_weights = nn.Parameter(torch.ones(3) * 0.33)

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast_ICML(self, x_enc, x_mark_enc, batch_y, x_mark_dec):
        # Normalization from Non-stationary Transformer    / 可以试一下revin  /或者在当前这个模块下加个W学一下
        _, _, N = x_enc.shape
        x_enc_3 = x_enc
        _, x_enc_2 = self.decomp(x_enc_3)
        _, x_enc_1 = self.decomp(x_enc_2)

        _ = self.revin_layer_1(x_enc_1, 'norm')
        _ = self.revin_layer_2(x_enc_2, 'norm')
        _ = self.revin_layer_3(x_enc_3, 'norm')

        # Embedding
        enc_emb = self.enc_embedding_1(x_enc, None)

        # level 1:
        enc_out_1 = self.encoder_ICML_1(enc_emb)
        enc_emb_1 = self.decoder_ICML_1(enc_out_1)
        dec_out_1 = self.revin_layer_1(enc_emb_1.permute(0, 2, 1), 'denorm')

        # level 2:
        enc_out_2 = torch.cat([self.enc_embedding_2(dec_out_1, None), self.mapping_ICML_12(enc_emb)], -1)
        enc_out_2 = self.encoder_ICML_2(enc_out_2)
        enc_emb_2 = self.decoder_ICML_2(enc_out_2)
        dec_out_2 = self.revin_layer_2(enc_emb_2.permute(0, 2, 1), 'denorm')

        # level 3:
        enc_out_3 = torch.cat([self.enc_embedding_3(dec_out_2, None), self.mapping_ICML_23(enc_emb)], -1)
        enc_out_3 = self.encoder_ICML_3(enc_out_3)
        enc_emb_3 = self.decoder_ICML_3(enc_out_3)
        dec_out_3 = self.revin_layer_3(enc_emb_3.permute(0, 2, 1), 'denorm')

        # # level 4
        # enc_out_4, means_4, stdev_4 = self.non_norm(dec_out_3)
        # enc_out_4 = self.enc_embedding_4(enc_out_4, None)
        # enc_out_4 = self.encoder_ICML_4(enc_out_4)
        # dec_out_4 = self.decoder_ICML_4(enc_out_4 + self.mapping_ICML_34(enc_emb)[:, :N, :])
        # dec_out_4 = self.non_denorm(dec_out_4.permute(0, 2, 1)[:, :, :N], means_4, stdev_4)

        # 不merge，通过调节loss实现scale 学习
        return dec_out_1, dec_out_2, dec_out_3

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, batch_y, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_1, dec_out_2, dec_out_3 = self.forecast_ICML(x_enc, x_mark_enc, batch_y, x_mark_dec)
            return dec_out_1, dec_out_2, dec_out_3
            # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, batch_y, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
