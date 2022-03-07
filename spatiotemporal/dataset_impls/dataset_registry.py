from functools import partial

from .registry import register_with_dictionary
from .weatherbench import WeatherBench

DATASET_IMPLEMENTATION_REGISTRY = {}
register = partial(register_with_dictionary, DATASET_IMPLEMENTATION_REGISTRY)


register(WeatherBench)

def get_dataset_implementation(name):
    return DATASET_IMPLEMENTATION_REGISTRY[name]
