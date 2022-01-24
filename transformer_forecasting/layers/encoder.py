import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.normalization import LayerNorm
from layers.series_decomposition import SeriesDecomp
from layers.autocorrelation import AutoCorrelationLayer



class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self,
                 n_attn_layers, # Number of Attention Layers
                 n_heads, # Number of attention heads
                 d_model, # Dimension of model features
                 d_ff, # Dimension of Feed Forward Layer
                 ma_window_size, # Moving Average Window Size
                 attention_dropout, # Dropout for Attention Layers
                 activation, # Activation Function,
                 factor, # Attention Factor,
                 output_attention, # Bool whether to output attention
                 norm_layer # Bool whether to include layer norm layer
                 ):

        super(Encoder, self).__init__()

        self.n_attn_layers = n_attn_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.ma_window_size = ma_window_size
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.factor = factor
        self.norm_layer = LayerNorm(d_model) if norm_layer else None

        encoder_layers = []
        for l in range(n_attn_layers):
            encoder_layer = EncoderLayer(n_heads,
                                         d_model,
                                         d_ff,
                                         ma_window_size,
                                         attention_dropout,
                                         activation,
                                         factor,
                                         output_attention
                                         )

            encoder_layers.append(encoder_layer)

        self.attn_layers = nn.ModuleList(encoder_layers)

    def forward(self, x, attn_mask=None):
        attns = []

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self,
                 n_heads,
                 d_model,
                 d_ff,
                 ma_window_size,
                 dropout,
                 activation,
                 factor,
                 output_attention
                 ):

        super(EncoderLayer, self).__init__()
        self.attention = AutoCorrelationLayer(False, n_heads, d_model, factor, dropout, output_attention)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(ma_window_size)
        self.decomp2 = SeriesDecomp(ma_window_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn