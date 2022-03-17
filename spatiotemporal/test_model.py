import torch
import torch.nn.functional as F
import pytest

from models.convlstm import ConvLSTMLayer, ConvLSTMEncoder
from models.dilated_net import DilatedBlock, DilatedEncoder
from models.downsample import Downsample2d, ParameterizedDownsample2d, CropAndTile
from models.lead_time_system import LeadTimeMLPSystem
from models.resnet import Block, ResNet
from models.metnet2 import MetNet2

from test_utils.fixtures import get_device


@torch.no_grad()
def test_convlstm_layer(get_device):
    """Test a single convlstm laye on one timestep
    Args:
    """
    args = [3, 128, 1024]  # in_channels  # hidden_channels  # lt_features
    layer = ConvLSTMLayer(*args).to(get_device)

    dummy_input = torch.randn(16, 3, 64, 64).to(get_device)
    dummy_hidden = torch.randn(16, 128, 64, 64).to(get_device)
    dummy_cell = torch.randn(16, 128, 64, 64).to(get_device)
    scale_bias = torch.randn(16, 1024).to(get_device)

    output = layer(
        torch.cat((dummy_input, dummy_hidden, dummy_cell), dim=-3), scale_bias
    )

    assert output.shape == (16, 128 + 128, 64, 64)


@torch.no_grad()
def test_convlstm_encoder(get_device):
    """Test the full convlstm network for a number of timesteps
    Args:
    """
    args = [
        3,  # in_channels
        [128] * 5,  # network_hidden_channels   (5 layers, each with 128 channels)
        1024,  # lt_features
    ]
    encoder = ConvLSTMEncoder(*args).to(get_device)

    dummy_input = torch.randn(16, 12, 3, 64, 64).to(get_device)  # 12 timesteps
    scale_bias = torch.randn(16, 1024).to(get_device)

    output = encoder(dummy_input, scale_bias)

    assert output.shape == (16, 128, 64, 64)


@torch.no_grad()
def test_dilated_block(get_device):
    """Test the dilated residual block
    Args:
    """
    args = [
        128,  # in_channels
        256,  # hidden_channels
        64,  # h_dim
        64,  # w_dim
        4,  # dilation
        1024,  # lt_features
    ]
    block = DilatedBlock(*args).to(get_device)

    dummy_input = torch.randn(16, 128, 64, 64).to(get_device)
    scale_bias = torch.randn(16, 1024).to(get_device)

    out, scale_bias_out = block(dummy_input, scale_bias)

    assert out.shape == (16, 256, 64, 64)
    assert torch.equal(scale_bias_out, scale_bias)


@torch.no_grad()
def test_dilated_encoder(get_device):
    """Test the entire dilated encoder stack
    Args:
    """
    args = [
        128,  # in_channels
        256,  # hidden_channels
        [1, 2, 4, 8, 16, 32, 64],  # dilations
        64,  # h_dim
        64,  # w_dim
        1024,  # lt_features
    ]
    encoder = DilatedEncoder(*args).to(get_device)

    dummy_input = torch.randn(16, 128, 64, 64).to(get_device)
    scale_bias = torch.randn(16, 1024).to(get_device)

    out, scale_bias_out = encoder(dummy_input, scale_bias)

    assert out.shape == (16, 256, 64, 64)
    assert torch.equal(scale_bias_out, scale_bias)


@torch.no_grad()
def test_downsample2d(get_device):
    args = ["maxpool", (4, 4), 0]  # pool_type  # kernel_size  # padding
    down = Downsample2d(*args).to(get_device)

    dummy_input = torch.randn(16, 12, 3, 64, 64).to(get_device)

    output = down(dummy_input)

    assert output.shape == (16, 12, 3, 16, 16)


@torch.no_grad()
def test_param_downsample2d(get_device):
    args = [3, (2, 2), 2, 0]  # hidden_channels  # kernel_size  # stride  # padding
    down = ParameterizedDownsample2d(*args).to(get_device)

    dummy_input = torch.randn(16, 12, 3, 64, 64).to(get_device)
    
    output = down(dummy_input)

    assert output.shape == (16, 12, 3, 32, 32)


def test_crop_tile(get_device):
    args = [32, 32, (5, 3)]  # crop_height  # crop_width  # tile
    croptile = CropAndTile(*args).to(get_device)

    dummy_input = torch.randn(16, 128, 32, 32).to(get_device)

    output = croptile(dummy_input)

    assert output.shape == (16, 128, 160, 96)


@torch.no_grad()
def test_lead_time_system(get_device):
    """Test the lead time system
    Args:
    """
    args = [
        12,  # lead_timesteps
        5,  # master_layers
        1024,  # master_out_features
        True,  # master_bias
    ]
    lts = LeadTimeMLPSystem(*args).to(get_device)

    num_classes = 12
    input_onehots = F.one_hot(
        torch.randint(low=0, high=12, size=(16,)), num_classes=num_classes
    ).to(get_device)

    output = lts(input_onehots.float())

    assert output.shape == (16, 1024)


@torch.no_grad()
def test_block(get_device):
    """Test the simple resnet conv blocks"""
    args = [
        3,  # in_channels
        32,  # out_channels
        64,  # h_dim
        64,  # w_dim
        1024,  # lt_features
    ]
    block = Block(*args).to(get_device)

    test_in = torch.randn(16, 3, 64, 64).to(get_device)
    scale_bias = torch.randn(16, 1024).to(get_device)

    out, scale_bias_out = block(test_in, scale_bias)

    assert out.shape == (16, 32, 64, 64)
    assert torch.equal(scale_bias_out, scale_bias)


@torch.no_grad()
def test_resnet(get_device):
    """Test the entire resnet stack"""
    args = [
        3,  # in_channels
        64,  # hidden_channels
        1,  # out_channels
        64,  # h_dim
        64,  # w_dim
        8,  # num_layers
        1024,  # lt_features
    ]
    resnet = ResNet(*args).to(get_device)

    test_in = torch.randn(16, 3, 64, 64).to(get_device)
    scale_bias = torch.randn(16, 1024).to(get_device)

    out, scale_bias_out = resnet(test_in, scale_bias)

    assert out.shape == (16, 1, 64, 64)
    assert torch.equal(scale_bias_out, scale_bias)

@torch.no_grad()
def test_metnet2(get_device):
    """Test MetNet2 model outputs
    Args:
    """
    args = [
        2e-5,
        0.1,
        12,
        2,
        [128, 128, 128],
        32,
        32,
        4,
        256,
        12,
        2,
        1024,
        True,
        1,
        8
    ]
    metnet = MetNet2(*args).to(get_device)

    dummy_input = torch.randn(16, 12, 2, 64, 64).to(get_device)
    num_classes = 12
    input_onehots = F.one_hot(
        torch.randint(low=0, high=12, size=(16,)), num_classes=num_classes
    ).to(get_device)


    output = metnet(dummy_input, input_onehots.float())

    assert output.shape == (16, 1, 32, 32)

