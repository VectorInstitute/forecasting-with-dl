import torch
import torch.nn as nn

from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.series_decomposition import SeriesDecomp
from layers.embedding import DataEmbedding_wo_pos

class AutoFormer(nn.Module):

    def __init__(self,
                 seq_len=96, # Sequence Length
                 label_len=48, # Start Token Length
                 pred_len=96, # Prediction Sequence Length
                 ma_window_size=25, # Kernel Size of moving average
                 enc_in=8, # Encoder Input Size
                 dec_in=8, # Decoder Input Size
                 d_model=512, # Dimension of Model
                 d_ff=2048, # Dimension of FCN
                 n_heads=8, # Number of Attention Heads
                 embed="timeF", # Time features encoding
                 freq="d", # Frequency of time feature encodings
                 dropout=0.05, # Dropout rate of network
                 factor=1, # Attention Factor
                 activation="gelu", # Activation function
                 e_layers=2, # Number of encoder layers
                 d_layers=1, # Number of decoder layers
                 c_out=8, # Output Size
                 output_attention=False
                 ):

        super(AutoFormer, self).__init__()
        # Set Attributes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        self.c_out = c_out

        self.decomp = SeriesDecomp(ma_window_size)

        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Build Encoder
        self.encoder = Encoder(e_layers,
                               n_heads,
                               d_model,
                               d_ff,
                               ma_window_size,
                               dropout,
                               activation,
                               factor,
                               output_attention,
                               norm_layer=True
                               )

        self.decoder = Decoder(
            d_layers,
            n_heads,
            d_model,
            d_ff,
            c_out,
            ma_window_size,
            dropout,
            activation,
            factor,
            norm_layer=True,
            projection=True
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                   trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
