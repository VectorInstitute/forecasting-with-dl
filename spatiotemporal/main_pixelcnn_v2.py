import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import wandb

from models.ar_decoder import GatedPixelCNN
from dataset_impls.dataset_registry import get_dataset_implementation
from utils.visualization_tools import check_grads, count_parameters, tensor_to_images


def boolean_string(s):
    if s not in {"True", "False"}:
        raise ValueError("Not a proper boolean string")
    else:
        return s == "True"


def prepare_args():
    parser = argparse.ArgumentParser(description="Args for training")
    # Dataset params
    parser.add_argument("--dset_path",
                        type=str,
                        default="/h/mchoi/SpatioStuff/dataset_impls/datasets")
    parser.add_argument("--dset", type=str, default="CustomMNIST")

    # Training params
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--rdm_seed", type=int, default=42069)
    parser.add_argument("--use_wandb", type=boolean_string, default=False)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/h/mchoi/SpatioStuff/models/checkpoints/pixelcnn.pt")
    parser.add_argument("--sample_output_path",
                        type=str,
                        default="/h/mchoi/SpatioStuff/samples_mnist/")

    # Model params
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=7)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0)

    args = parser.parse_args()
    return args


def save(model: nn.Module, optim: torch.optim.Optimizer, args) -> None:
    """Save the model and optimizer states
     """
    torch.save({
        "pixelcnn": model.state_dict(),
        "optim": optim.state_dict(),
    }, args.checkpoint_path)


def load(args, model_args: list, device: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Load the model and optimizer states
     """
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError("Checkpoint file doesn't exist!")
    model = GatedPixelCNN(*model_args)
    model.to(device)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=args.lr,
                             weight_decay=args.wd)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["pixelcnn"])
    optim.load_state_dict(checkpoint["optim"])

    return model, optim


def setup_dataloaders(args) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Setup transforms
    if "CustomCIFAR10" in args.dset:
        mean_std = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
    elif "CustomMNIST" in args.dset:
        mean_std = transforms.Normalize((0.1307, ), (0.3081, ))
    transform_list = transforms.Compose([
        transforms.ToTensor(),
        #mean_std,
    ])
    # Setup dataset
    train_dset = get_dataset_implementation(args.dset)(
        args.dset_path,
        train=True,
        transform=transform_list,
    )
    val_dset = get_dataset_implementation(args.dset)(
        args.dset_path,
        train=False,
        transform=transform_list,
    )
    # Setup dataloaders
    train_dl = torch.utils.data.DataLoader(dataset=train_dset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
    val_dl = torch.utils.data.DataLoader(dataset=val_dset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True)
    return train_dl, val_dl


def setup_model_args(args) -> list:
    if "MNIST" in args.dset:
        model_args = [
            28, 28, args.in_channels, args.out_channels, args.kernel_size, 1, args.n_layers,
            args.dropout
        ]
    if "CIFAR10" in args.dset:
        model_args = [
            32, 32, args.in_channels, args.out_channels, args.kernel_size, 1, args.n_layers,
            args.dropout
        ]
    return model_args


def setup_wandb(args) -> None:
    project_name = f"GatedPixelCNN on {args.dset}"
    wandb.init(project=project_name, entity="mchoi")
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size
    }


def setup_model(args, model_args: list, device: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
    if os.path.isfile(
            args.checkpoint_path) and "None" not in args.checkpoint_path:
        model, optim = load(args, model_args, device)
        print(f"Loaded model from checkpoint: {args.checkpoint_path}")
    else:
        model = GatedPixelCNN(*model_args)
        model.to(device)
        optim = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    return model, optim


def train(args, device: str, train_dl: torch.utils.data.DataLoader,
          val_dl: torch.utils.data.DataLoader, model: nn.Module,
          optim: torch.optim.Optimizer) -> None:
    loss_fn = nn.CrossEntropyLoss()
    # Train/val loop
    from tqdm import tqdm
    train_loss = []
    for i in tqdm(range(args.num_epochs)):
        model.train()
        epoch_loss = 0

        #Training epoch
        for example, _ in tqdm(train_dl):
            example = example.to(device)
            optim.zero_grad()
            output = model(example)
            loss = loss_fn(output, (example * 7).long())
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            if args.use_wandb is True:
                wandb.log({"example loss": loss.item()})
            else:
                print(f"Example loss: {loss.item()}")

        epoch_loss /= len(train_dl)
        train_loss.append(epoch_loss)

        if "None" not in args.checkpoint_path:
            save(model, optim, args)

        if args.use_wandb is True:
            wandb.log({"epoch loss": epoch_loss})
        else:
            print(f"Epoch loss: {epoch_loss}")

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for example, _ in tqdm(val_dl):
                example = example.to(device)
                output = model(example)
                loss = loss_fn(output, (example * 7).long())
                val_loss += loss.item()
            val_loss /= len(val_dl)
            if args.use_wandb is True:
                wandb.log({"val_loss": val_loss})
            else:
                print(f"Validation loss: {val_loss}")
            sample(args, model, 24, example.shape[2], example.shape[3], i)


def sample(args, model: nn.Module, num_samples: int, h_dim: int,
           w_dim: int, epoch: int) -> None:
    if "None" not in args.sample_output_path:
        samples = model.sample(num_samples, (args.in_channels, h_dim, w_dim))
        samples_grid = torchvision.utils.make_grid(samples, normalize=True, value_range=(0, 7))
        torchvision.utils.save_image(samples_grid, args.sample_output_path + f"test_sample_epoch{epoch}.png")


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.rdm_seed)

    if args.use_wandb is True:
        setup_wandb(args)

    print(args)

    train_dl, val_dl = setup_dataloaders(args)

    model_args = setup_model_args(args)

    model, optim = setup_model(args, model_args, device)

    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")

    #sample(args, model, 24, 28, 28, 0)
    train(args, device, train_dl, val_dl, model, optim)


if __name__ == "__main__":
    args = prepare_args()
    main(args)
