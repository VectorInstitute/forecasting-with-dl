import os
import argparse
from argparse import Namespace
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torchvision
from torchvision import transforms
import wandb

from dataset_impls.dataset_registry import get_dataset_implementation
from models.convlstm_v2 import ConvLSTMForecastingNet
from utils.visualization_tools import plot_grad_flow, check_grads, tensor_to_gif, count_parameters


def boolean_string(s: str) -> bool:
    if s not in {"True", "False"}:
        raise ValueError("Not a proper boolean string")
    else:
        return s == "True"


def prepare_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Args for training")
    # Dataset params
    parser.add_argument("--dset_path",
                        type=str,
                        default="./dataset_impls/datasets/moving_mnist.npy")
    parser.add_argument("--dset", type=str, default="MovingMnist")

    # Training params
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--rdm_seed", type=int, default=42069)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/h/mchoi/SpatioStuff/models/checkpoints/lstm_default_chk.pt")
    parser.add_argument(
        "--sample_output_path",
        type=str,
        default="/h/mchoi/SpatioStuff/samples_convlstm_default/")
    parser.add_argument("--loss_fn", type=str, default="MSELoss")

    # Model params
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--hidden_channels", nargs="+")

    # Wandb params
    parser.add_argument("--use_wandb", type=boolean_string, default=False)
    parser.add_argument("--project_name", type=str, default="Default_project")

    args = parser.parse_args()
    return args


def save(model: nn.Module, optim: Optimizer, args: Namespace) -> None:
    """Save the model and optimizer states
    """
    torch.save({
        "convlstm": model.state_dict(),
        "optim": optim.state_dict(),
    }, args.checkpoint_path)


def load(args: Namespace, model_args: list,
         device: str) -> Tuple[nn.Module, Optimizer]:
    """Load the model and optimizer states
    """
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError("Checkpoint file doesn't exist!")
    model = ConvLSTMForecastingNet(*model_args)
    model.to(device)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=args.lr,
                             weight_decay=args.wd)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["convlstm"])
    optim.load_state_dict(checkpoint["optim"])

    return model, optim


def setup_dataloaders(
        args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Dataset
    transform = transforms.Compose([
        #transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_dset = get_dataset_implementation(args.dset)(args.dset_path,
                                                       split="train",
                                                       transform=transform)
    val_dset = get_dataset_implementation(args.dset)(args.dset_path,
                                                     split="val",
                                                     transform=transform)
    test_dset = get_dataset_implementation(args.dset)(args.dset_path,
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


def setup_model_args(args: Namespace) -> list:
    #if "WeatherBench" in args.dset:
    #    resolution = [128, 256]
    #elif "MovingMnist" in args.dset:
    #    resolution = [64, 64]
    hidden_channels = tuple(map(int, args.hidden_channels))
    model_args = [args.in_channels, args.seq_len, hidden_channels]

    return model_args


def setup_model(args: Namespace, model_args: list,
                device: str) -> Tuple[nn.Module, Optimizer]:
    if os.path.isfile(
            args.checkpoint_path) and "None" not in args.checkpoint_path:
        model, optim = load(args, model_args, device)
        print(f"Loaded from checkpoint: {args.checkpoint_path}")
    else:
        model = ConvLSTMForecastingNet(*model_args)
        model.to(device)  # Must put model on device before feeding into optim
        optim = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    return model, optim


def setup_wandb(args: Namespace) -> None:
    if args.use_wandb is True:
        wandb.init(project=f"{args.project_name} on {args.dset}", entity="mchoi")
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        }


def train(args: Namespace, device: str, train_dl: DataLoader,
          val_dl: DataLoader, test_dl: DataLoader, model: nn.Module,
          optim: Optimizer) -> None:

    # Loss function
    if args.loss_fn == "MSELoss":
        loss_fn = nn.MSELoss()
    elif args.loss_fn == "BCELoss":
        loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")

    # Train/val loop
    from tqdm import tqdm
    train_loss = []
    for i in tqdm(range(args.num_epochs)):
        model.train()
        epoch_loss = 0

        # Training epoch
        for example, target in tqdm(train_dl):
            example = example.to(device)
            target = target.to(device)
            optim.zero_grad()
            preds, _ = model(example)
            loss = loss_fn(preds, target)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dl)
        train_loss.append(epoch_loss)

        # Checkpoint model
        if "None" not in args.checkpoint_path:
            save(model, optim, args)

        # Log to wandb
        if args.use_wandb is True:
            wandb.log({"loss": epoch_loss})
        else:
            print(f"Epoch loss: {epoch_loss}")

        # Validation post-epoch
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for example, target in val_dl:
                example = example.to(device)
                target = target.to(device)
                preds, _ = model(example)
                loss = loss_fn(preds, target)
                val_loss += loss.item()
            val_loss /= len(val_dl)
            if args.use_wandb is True:
                wandb.log({"val_loss": val_loss})
            else:
                print(f"Validation loss: {val_loss}")
            if "None" not in args.sample_output_path:
                sample(args, example, target, preds, i)

    # Final test epoch
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for example, target in test_dl:
            example = example.to(device)
            target = target.to(device)
            preds, _ = model(example)
            loss = loss_fn(preds, target)
            test_loss += loss.item()
        test_loss /= len(test_dl)
        if args.use_wandb is True:
            wandb.log({"Test loss: {test_loss}"})
        else:
            print(f"Test loss: {test_loss}")


def sample(args: Namespace, example: Tensor, target: Tensor, preds: Tensor,
           epoch: int) -> None:

    if not os.path.isdir(args.sample_output_path):
        os.mkdir(args.sample_output_path)

    # Sample tensor from last batch
    sample_tensor = torch.cat((example[0], target[0], preds[0]), dim=0)
    samples = torchvision.utils.make_grid(sample_tensor.detach().cpu(),
                                          nrow=args.seq_len,
                                          normalize=True)
    torchvision.utils.save_image(
        samples, args.sample_output_path + f"{args.dset}_epoch{epoch}.png")


def main(args: Namespace) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.rdm_seed)

    if args.use_wandb is True:
        setup_wandb(args)

    print(args)

    train_dl, val_dl, test_dl = setup_dataloaders(args)

    model_args = setup_model_args(args)

    model, optim = setup_model(args, model_args, device)

    train(args, device, train_dl, val_dl, test_dl, model, optim)


if __name__ == "__main__":
    args = prepare_args()
    main(args)
