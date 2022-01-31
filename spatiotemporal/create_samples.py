import os

import torch
import torch.nn as nn
import torchvision

from models.metnet2_pl import MetNet2
from dataset_impls.moving_mnist import MovingMnist


def main():
    checkpoint_file = "/h/mchoi/SpatioStuff/models/checkpoints/metnet2/metnet2_div0.6-v2.ckpt"
    if os.path.isfile(checkpoint_file) is False:
        raise Exception("File doesn't exist")

    model = MetNet2.load_from_checkpoint(checkpoint_file)
    dset = MovingMnist(src_file="/h/mchoi/SpatioStuff/dataset_impls/datasets/moving_mnist.npy", split="test", div=1.0)

    for i in range(10):
        leadtime_vecs = nn.functional.one_hot(torch.arange(10), num_classes=10)
        frames = torch.zeros(20, 1, 64, 64)
        example, target, _ = dset[i * 10]
        for j, leadtime_vec in enumerate(leadtime_vecs):
            frame = model(example.unsqueeze(0), leadtime_vec.unsqueeze(0).unsqueeze(1).float())
            frames[j + 10, :, :, :] = frame
        frames[:10, :, :, :] = example
        grid_sample = torchvision.utils.make_grid(frames, nrow=10)
        torchvision.utils.save_image(grid_sample, f"SAMPLE{i}.png")



if __name__ == "__main__":
    main()
