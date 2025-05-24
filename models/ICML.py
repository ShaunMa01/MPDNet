import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.Autoformer_EncDec import series_decomp
from layers.Revin import RevIN

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = torch.device('cuda:0')
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        self.revin_layer_1 = RevIN(configs.enc_in)
        self.revin_layer_2 = RevIN(configs.enc_in)
        self.revin_layer_3 = RevIN(configs.enc_in)

        # Embedding
        self.enc_embedding_1 = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.enc_embedding_2 = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.enc_embedding_3 = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # self.enc_embedding_4 = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 后边可以弄成超参数
        self.encoder_ICML_1 = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.decoder_ICML_1 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.con_ICML_12 = nn.Linear(configs.pred_len, configs.d_model, bias=True)
        self.encoder_ICML_2 = nn.Linear(configs.d_model*2, configs.d_model, bias=True)
        self.mapping_ICML_12 = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.decoder_ICML_2 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.con_ICML_23 = nn.Linear(configs.pred_len, configs.d_model, bias=True)
        self.encoder_ICML_3 = nn.Linear(configs.d_model*2, configs.d_model, bias=True)
        self.mapping_ICML_23 = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.decoder_ICML_3 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # self.encoder_ICML_4 = nn.Linear(configs.d_model, configs.d_model, bias=True)
        # self.mapping_ICML_34 = nn.Linear(configs.d_model, configs.d_model, bias=True)
        # self.decoder_ICML_4 = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.merge_weights = nn.Parameter(torch.ones(3)*0.33)


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

    # def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #     x_enc /= stdev

    #     _, _, N = x_enc.shape

    #     # Embedding
    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)

    #     dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
    #     # De-Normalization from Non-stationary Transformer
    #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    #     return dec_out
    def non_norm(self,x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        return x_enc,means,stdev

    def non_denorm(self,dec_out,means,stdev):
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
    
    # 当前这一版没有非线性处理
    # def forecast_ICML(self, x_enc, x_mark_enc, batch_y, x_mark_dec):
    #     # Normalization from Non-stationary Transformer    / 可以试一下revin  /或者在当前这个模块下加个W学一下
    #     x_enc_3 = x_enc
    #     _, x_enc_2 = self.decomp(x_enc)
    #     _, x_enc_1 = self.decomp(x_enc_2)
    #
    #     x_enc_r = self.revin_layer(x_enc_1, 'norm')
    #     _, _, N = x_enc.shape
    #     # Embedding
    #     enc_emb = self.enc_embedding_1(x_enc_1, x_mark_enc)
    #
    #     # level 1:
    #     enc_out_1 = self.encoder_ICML_1(enc_emb)
    #     dec_out_1 = self.decoder_ICML_1(enc_out_1)
    #     dec_out_1 = self.non_denorm(dec_out_1.permute(0, 2, 1)[:, :, :N],means_1,stdev_1)
    #
    #     # level 2:
    #     enc_out_2, means_2, stdev_2 = self.non_norm(dec_out_1)
    #     enc_out_2 = self.enc_embedding_2(enc_out_2,None)
    #     enc_out_2 = self.encoder_ICML_2(enc_out_2)
    #     dec_out_2 = self.decoder_ICML_2(enc_out_2 + self.mapping_ICML_12(enc_emb)[:,:N,:])
    #     dec_out_2 = self.non_denorm(dec_out_2.permute(0, 2, 1)[:, :, :N],means_2,stdev_2)
    #
    #     # level 3:
    #     enc_out_3, means_3, stdev_3 = self.non_norm(dec_out_2)
    #     enc_out_3 = self.enc_embedding_3(enc_out_3,None)
    #     enc_out_3 = self.encoder_ICML_3(enc_out_3)
    #     dec_out_3 = self.decoder_ICML_3(enc_out_3 + self.mapping_ICML_23(enc_emb)[:,:N,:])
    #     dec_out_3 = self.non_denorm(dec_out_3.permute(0, 2, 1)[:, :, :N],means_3,stdev_3)
    #     x_out = self.revin_layer(x_out, 'denorm')
    #
    #     # # level 4
    #     # enc_out_4, means_4, stdev_4 = self.non_norm(dec_out_3)
    #     # enc_out_4 = self.enc_embedding_4(enc_out_4, None)
    #     # enc_out_4 = self.encoder_ICML_4(enc_out_4)
    #     # dec_out_4 = self.decoder_ICML_4(enc_out_4 + self.mapping_ICML_34(enc_emb)[:, :N, :])
    #     # dec_out_4 = self.non_denorm(dec_out_4.permute(0, 2, 1)[:, :, :N], means_4, stdev_4)
    #
    #     # 不merge，通过调节loss实现scale 学习
    #     return dec_out_1,dec_out_2,dec_out_3

    # 0.300/0.350  h2
    # def forecast_ICML(self, x_enc, x_mark_enc, batch_y, x_mark_dec):
    #     # Normalization from Non-stationary Transformer    / 可以试一下revin  /或者在当前这个模块下加个W学一下
    #
    #     x_enc_1, means_1, stdev_1 = self.non_norm(x_enc)
    #     _, _, N = x_enc.shape
    #     # Embedding
    #     enc_emb = self.enc_embedding_1(x_enc_1, x_mark_enc)
    #
    #     # level 1:
    #     enc_out_1 = self.encoder_ICML_1(enc_emb)
    #     dec_out_1 = self.decoder_ICML_1(enc_out_1)
    #     dec_out_1 = self.non_denorm(dec_out_1.permute(0, 2, 1)[:, :, :N], means_1, stdev_1)
    #
    #     # level 2:
    #     enc_out_2, means_2, stdev_2 = self.non_norm(dec_out_1)
    #     enc_out_2 = self.enc_embedding_2(enc_out_2, None)
    #     enc_out_2 = self.encoder_ICML_2(enc_out_2)
    #     dec_out_2 = self.decoder_ICML_2(enc_out_2 + self.mapping_ICML_12(enc_emb)[:, :N, :])
    #     dec_out_2 = self.non_denorm(dec_out_2.permute(0, 2, 1)[:, :, :N], means_2, stdev_2)
    #
    #     # level 3:
    #     enc_out_3, means_3, stdev_3 = self.non_norm(dec_out_2)
    #     enc_out_3 = self.enc_embedding_3(enc_out_3, None)
    #     enc_out_3 = self.encoder_ICML_3(enc_out_3)
    #     dec_out_3 = self.decoder_ICML_3(enc_out_3 + self.mapping_ICML_23(enc_emb)[:, :N, :])
    #     dec_out_3 = self.non_denorm(dec_out_3.permute(0, 2, 1)[:, :, :N], means_3, stdev_3)
    #
    #     # # level 4
    #     # enc_out_4, means_4, stdev_4 = self.non_norm(dec_out_3)
    #     # enc_out_4 = self.enc_embedding_4(enc_out_4, None)
    #     # enc_out_4 = self.encoder_ICML_4(enc_out_4)
    #     # dec_out_4 = self.decoder_ICML_4(enc_out_4 + self.mapping_ICML_34(enc_emb)[:, :N, :])
    #     # dec_out_4 = self.non_denorm(dec_out_4.permute(0, 2, 1)[:, :, :N], means_4, stdev_4)
    #
    #     # 不merge，通过调节loss实现scale 学习
    #     return dec_out_1, dec_out_2, dec_out_3

    # 0.359 h2
    # def forecast_ICML(self, x_enc, x_mark_enc, batch_y, x_mark_dec):
    #     # Normalization from Non-stationary Transformer    / 可以试一下revin  /或者在当前这个模块下加个W学一下
    #     _,_,N = x_enc.shape
    #     x_enc_3 = x_enc
    #     _, x_enc_2 = self.decomp(x_enc_3)
    #     _, x_enc_1 = self.decomp(x_enc_2)
    #
    #     _ = self.revin_layer_1(x_enc_1, 'norm')
    #     _ = self.revin_layer_2(x_enc_2, 'norm')
    #     _ = self.revin_layer_3(x_enc_3, 'norm')
    #
    #     # Embedding
    #     enc_emb = self.enc_embedding_1(x_enc, None)
    #
    #     # level 1:
    #     enc_out_1 = self.encoder_ICML_1(enc_emb)
    #     enc_emb_1 = self.decoder_ICML_1(enc_out_1)
    #     dec_out_1 = self.revin_layer_1(enc_emb_1.permute(0, 2, 1), 'denorm')
    #
    #     # level 2:
    #     enc_out_2 = self.con_ICML_12(dec_out_1.permute(0, 2, 1)) + self.mapping_ICML_12(enc_emb)
    #     enc_out_2 = self.encoder_ICML_2(enc_out_2)
    #     enc_emb_2 = self.decoder_ICML_2(enc_out_2)
    #     dec_out_2 = self.revin_layer_2(enc_emb_2.permute(0, 2, 1), 'denorm')
    #
    #     # level 3:
    #     enc_out_3 = self.con_ICML_23(dec_out_2.permute(0, 2, 1)) + self.mapping_ICML_23(enc_emb)
    #     enc_out_3 = self.encoder_ICML_3(enc_out_3)
    #     enc_emb_3 = self.decoder_ICML_3(enc_out_3)
    #     dec_out_3 = self.revin_layer_3(enc_emb_3.permute(0, 2, 1), 'denorm')
    #
    #     # # level 4
    #     # enc_out_4, means_4, stdev_4 = self.non_norm(dec_out_3)
    #     # enc_out_4 = self.enc_embedding_4(enc_out_4, None)
    #     # enc_out_4 = self.encoder_ICML_4(enc_out_4)
    #     # dec_out_4 = self.decoder_ICML_4(enc_out_4 + self.mapping_ICML_34(enc_emb)[:, :N, :])
    #     # dec_out_4 = self.non_denorm(dec_out_4.permute(0, 2, 1)[:, :, :N], means_4, stdev_4)
    #
    #     # 不merge，通过调节loss实现scale 学习
    #     return dec_out_1, dec_out_2, dec_out_3

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

    # def forward(self, x_enc, x_mark_enc, batch_y, x_mark_dec, mask=None):
    #     if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
    #         dec_out_1, dec_out_2, dec_out_3 = self.forecast_ICML(x_enc, x_mark_enc, batch_y, x_mark_dec)
    #         batch_y_3 = batch_y[:, -self.pred_len:, :].to(self.device)
    #
    #         _, batch_y_2 = self.decomp(batch_y_3)
    #         _, batch_y_1 = self.decomp(batch_y_2)
    #
    #
    #         loss = self.merge_weights[0] * torch.norm(batch_y_1 - dec_out_1, p=2) + \
    #                 self.merge_weights[1] * torch.norm(batch_y_2 - dec_out_2, p=2) + \
    #                 self.merge_weights[2] * torch.norm(batch_y_3 - dec_out_3, p=2)
    #
    #         return loss, dec_out_3
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
