"""PL VERSION OF MAIN FILE FOR METNET2"""

import os
import argparse
from argparse import Namespace
from typing import Tuple, List
import pprint

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchvision
from torchvision import transforms
import wandb

from models.metnet2_pl import MetNet2
from dataset_impls.dataset_registry import get_dataset_implementation
from models.multi_input_sequential import boolean_string
from utils.metnet2_image_callback import WandbImageCallback


def prepare_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Args for training")
    # Dataset params
    parser.add_argument("--dset_path",
                        type=str,
                        default="./dataset_impls/datasets/moving_mnist.npy")
    parser.add_argument("--dset", type=str, default="MovingMnist")
    parser.add_argument("--div", type=float, default=1.0)

    # Training params
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rdm_seed", type=int, default=42069)
    parser.add_argument("--checkpoint_path", type=str, default="/h/mchoi/SpatioStuff/models/checkpoints/")
    parser.add_argument("--checkpoint_filename", type=str, default="metnet2_default_chk")
    parser.add_argument("--sample_output_path", type=str, default="/h/mchoi/SpatioStuff/samples_metnet2_default/")

    # Wandb params
    parser.add_argument("--project_name", type=str, default="Default_project")

    # Get model specific args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MetNet2.add_model_specific_args(parser)
    args = parser.parse_args()
    return args

def setup_dataloaders(
        args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Dataset
    transform = transforms.Compose([
        #transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_dset = get_dataset_implementation(args.dset)(src=args.dset_path,
                                                       split="train",
                                                       transform=transform)
    val_dset = get_dataset_implementation(args.dset)(src=args.dset_path,
                                                     split="val",
                                                     transform=transform)
    test_dset = get_dataset_implementation(args.dset)(src=args.dset_path,
                                                      split="test",
                                                      transform=transform)
    train_dl = DataLoader(dataset=train_dset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=True)
    val_dl = DataLoader(dataset=val_dset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)
    test_dl = DataLoader(dataset=test_dset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=True)
    return train_dl, val_dl, test_dl

def get_val_samples(args, num_samples: int):
    val_dset = get_dataset_implementation(args.dset)(src=args.dset_path, 
                                                     split="val")
    xs = []
    targets = []
    leadtime_vecs = []
    for i in range(num_samples):
        x, target, leadtime_vec = val_dset[i]
        xs.append(x.unsqueeze(0))   # Unsqueeze sample batch dimension
        targets.append(target.unsqueeze(0))
        leadtime_vecs.append(leadtime_vec.unsqueeze(0))
    return torch.cat(xs, dim=0), torch.cat(targets, dim=0), torch.cat(leadtime_vecs, dim=0)

def setup_model(args: Namespace) -> pl.LightningModule:
    dict_args = vars(args)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict_args)

    checkpoint_file = args.checkpoint_path + f"{args.checkpoint_filename}_div{args.div}.ckpt"
    if os.path.isfile(checkpoint_file):
        model = MetNet2.load_from_checkpoint(checkpoint_file)
        print("*" * 40)
        print(f"Loaded from checkpoint: {checkpoint_file}")
        print("*" * 40)
    else:
        model = MetNet2(**dict_args)
    return model

def main(args: Namespace) -> None:
    pl.seed_everything(args.rdm_seed)

    # Init wandb
    wandb_logger = WandbLogger(project=args.project_name)

    # Init callbacks
    val_samples = get_val_samples(args, 1)
    callbacks = [LearningRateMonitor(logging_interval="step"),
                 ModelCheckpoint(dirpath=args.checkpoint_path, filename=f"{args.checkpoint_filename}_div{args.div}", every_n_train_steps=500),
                 WandbImageCallback(val_samples)
    ]

    # Get dataloaders
    train_dl, val_dl, test_dl = setup_dataloaders(args)

    # Init model
    model = setup_model(args)
    print(model)

    # Init trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    
    # Train
    trainer.fit(model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,)

if __name__ == "__main__":
    args = prepare_args()
    main(args)
