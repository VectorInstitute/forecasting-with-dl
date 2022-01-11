import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_grad_flow(named_parameters, save_fig):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    if save_fig is True:
        plt.savefig('/h/mchoi/SpatioStuff/grads.png')

def check_grads(named_parameters):
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                raise Exception(f"Grad for requires_grad=True parameter {n} is None!")

def tensor_to_gif(tensor, save_name, save_dir):
    tensor = (tensor - tensor.min()) * (255 / (tensor.max() - tensor.min()))
    unnormed_tensor = tensor.numpy().astype(np.uint8)  
    images = []
    for i, frame in enumerate(unnormed_tensor):
        img = Image.fromarray(frame[0]) # Get rid of channel dim
        images.append(img)
        images[i].save(save_dir + save_name, save_all=True, append_images=images, duration=500, loop=0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
