import torch
import torch.nn as nn


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        """Forward pass for sequential. This subclass allows for multiple arguments to be carried through a single nn.Sequential
        Args:
            inputs: Tuple containing inputs
        """
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def boolean_string(s: str) -> bool:
    """Function for proper processing of bools passed in through argparse + commandline"""
    if s not in {"True", "False"}:
        raise ValueError("Not a proper boolean string")
    else:
        return s == "True"
