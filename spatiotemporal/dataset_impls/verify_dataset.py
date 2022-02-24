import os

import torch
import torch.nn as nn
import torchvision

from moving_mnist import MovingMnist

print(f"Verifying dataset")

dset_train = MovingMnist(split="train", div=1.0)
dset_val = MovingMnist(split="val")
dset_test = MovingMnist(split="test")

for i in range(0, 100, 10):
    train_ex, train_target, _ = dset_train[i]
    val_ex, val_target, _ = dset_val[i]
    test_ex, test_target, _ = dset_test[i]
    grid_train = torchvision.utils.make_grid(train_ex, nrow=11)
    grid_val = torchvision.utils.make_grid(val_ex, nrow=11)
    grid_test = torchvision.utils.make_grid(test_ex, nrow=11)

    torchvision.utils.save_image(grid_train, f"GRID_TRAIN{i}.png")
    torchvision.utils.save_image(grid_val, f"GRID_VAL{i}.png")
    torchvision.utils.save_image(grid_test, f"GRID_TEST{i}.png")
