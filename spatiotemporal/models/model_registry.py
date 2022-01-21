from functools import partial

from registry import register_with_dictionary
from convlstm_v2 import ConvLSTMForecastingNet
from ar_decoder import GatedPixelCNN

MODEL_IMPLEMENTATION_REGISTRY = {}
register = partial(register_with_dictionary, MODEL_IMPLEMENTATION_REGISTRY)

register(ConvLSTMForecastingNet)
register(GatedPixelCNN)


def get_model_implementation(name):
    return MODEL_IMPLEMENTATION_REGISTRY[name]
