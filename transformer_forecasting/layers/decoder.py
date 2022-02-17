import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.normalization import LayerNorm
from layers.series_decomposition import SeriesDecomp
from layers.autocorrelation import AutoCorrelationLayer

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self,
                 n_heads,
                 d_model,
                 d_ff,
                 c_out,
                 ma_window_size,
                 dropout,
                 activation,
                 factor,
                 ):
        super(DecoderLayer, self).__init__()

        self.self_attention = AutoCorrelationLayer(True, n_heads, d_model, factor, dropout, False)
        self.cross_attention = AutoCorrelationLayer(False, n_heads, d_model, factor, dropout, False)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(ma_window_size)
        self.decomp2 = SeriesDecomp(ma_window_size)
        self.decomp3 = SeriesDecomp(ma_window_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self,
                 n_layers, # Number of Attention Layers
                 n_heads, # Number of attention heads
                 d_model, # Dimension of model features
                 d_ff, # Dimension of Feed Forward Layer
                 c_out, # Output dimensions
                 ma_window_size, # Moving Average Window Size
                 attention_dropout, # Dropout for Attention Layers
                 activation, # Activation Function,
                 factor, # Attention Factor,
                 norm_layer=False,
                 projection=False
                 ):
        super(Decoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.ma_window_size = ma_window_size
        self.activation = activation
        self.factor = factor

        self.norm_layer = LayerNorm(d_model) if norm_layer else None
        self.projection = nn.Linear(d_model, c_out, bias=True)

        decoder_layers = []
        for l in range(n_layers):
            decoder_layer = DecoderLayer(n_heads,
                                         d_model,
                                         d_ff,
                                         c_out,
                                         ma_window_size,
                                         attention_dropout,
                                         activation,
                                         factor,
                                         )

            decoder_layers.append(decoder_layer)

        self.layers = nn.ModuleList(decoder_layers)

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend
