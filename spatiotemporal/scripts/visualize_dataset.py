import os
import argparse

import numpy as np
from PIL import Image

from dataset_impls.moving_mnist import MovingMnist

def boolean_string(s):
    if s not in {"True", "False"}:
        raise ValueError("Not a proper boolean string")
    else:
        return s == "True"

def prepare_args():
    """Setup visualization args"""
    parser = argparse.ArgumentParser(description="Args for training")
    parser.add_argument("--dset_dir", type=str, default="./dataset_impls/datasets/moving_mnist.npy")
    parser.add_argument("--gif", type=boolean_string, default=False)
    parser.add_argument("--gif_save_dir", type=str, default="./dataset_impls/visualizations/")
    args_to_ret = parser.parse_args()
    return args_to_ret

def make_gif(dset, save_dir):
    for i in range(3):
        images = []
        frames = dset[i]
        for frame in frames:
            img = Image.fromarray(frame)
            images.append(img)
        images[i].save(save_dir + f"moving_mnist_ex{i}.gif", save_all=True, append_images=images, duration=500, loop=0)

def main(args):
    dset = MovingMnist(args.dset_dir, split="train")
    make_gif(dset, args.gif_save_dir)

if __name__ == "__main__":
    args = prepare_args()
    main(args)
