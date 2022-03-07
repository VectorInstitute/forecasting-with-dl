"""PL VERSION OF MAIN FILE FOR METNET2"""

import os
import argparse
from argparse import Namespace
from typing import Tuple, List, Dict, Any
import pprint
import time

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

from models.metnet2 import MetNet2
from dataset_impls.weatherbench_datamod import WeatherBenchDataModule
from dataset_impls.weatherbench import WeatherBench
from models.multi_input_sequential import boolean_string
from utils.metnet2_image_callback import WandbImageCallback


def prepare_args() -> Namespace:
    """Prepare commandline args"""
    parser = argparse.ArgumentParser(description="Args for training")
    # Data Module params
    parser.add_argument("--data_dir", type=str, default="/ssd003/projects/aieng/datasets/forecasting")
    parser.add_argument("--every_n", type=int, default=2)
    parser.add_argument("--n_obs", type=int, default=12)
    parser.add_argument("--n_target", type=int, default=12)
    parser.add_argument("--n_blackout", type=int,default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=10)

    # Training params
    parser.add_argument("--rdm_seed", type=int, default=42069)
    parser.add_argument("--checkpoint_path", type=str, default="/h/mchoi/spatiostuff/models/checkpoints/")
    parser.add_argument("--checkpoint_filename", type=str, default="metnet2_default_chk")

    # Wandb params
    parser.add_argument("--project_name", type=str, default="Default_project")

    # Get model specific args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MetNet2.add_model_specific_args(parser)
    args = parser.parse_args()
    return args

def get_val_samples(args, num_samples: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Retrieve validation samples from dataset
    Args:
        args (namespace): 
        num_samples (int): The number of validation samples (batch size) returned
    """
    val_dset = WeatherBench(src=args.data_dir, 
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

def setup_model(dict_args: Dict[str, Any]) -> pl.LightningModule:
    """Model creation/loading logic
    Args:
        args (namespace):
    """
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict_args)

    checkpoint_file = args.checkpoint_path + f"{args.checkpoint_filename}.ckpt"
    if os.path.isfile(checkpoint_file):
        print("*" * 40)
        print(f"Loaded from checkpoint: {checkpoint_file}")
        print("*" * 40)
        model = MetNet2.load_from_checkpoint(checkpoint_file)
    else:
        model = MetNet2(**dict_args)
    return model

def main(args: Namespace) -> None:
    pl.seed_everything(args.rdm_seed)

    # Init logger, True is default tensorboard logger
    logger = WandbLogger(project=args.project_name)  

    # Init callbacks
    val_samples = get_val_samples(args, 1)      
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(dirpath=args.checkpoint_path, 
                        filename=f"{args.checkpoint_filename}", 
                        every_n_train_steps=500),
        WandbImageCallback(val_samples),
    ]

    # Init model and dataset
    dict_args = vars(args)  
    weatherbench_dset = WeatherBenchDataModule(**dict_args)
    model = setup_model(dict_args)
    print(model)

    # Init trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
    )
    
    # Train
    trainer.fit(model, weatherbench_dset)

if __name__ == "__main__":
    args = prepare_args()
    main(args)
