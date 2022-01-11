import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import wandb

from dataset_impls.moving_mnist import MovingMnist
from models.convlstm_v2 import ConvLSTMForecastingNet
from utils.visualization_tools import plot_grad_flow, check_grads, tensor_to_gif, count_parameters

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
                        default="./dataset_impls/datasets/moving_mnist.npy")

    # Training params
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--rdm_seed", type=int, default=42069)
    parser.add_argument("--use_wandb", type=boolean_string, default=False)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/h/mchoi/SpatioStuff/models/checkpoints/lstm.pt")
    parser.add_argument("--sample_output_path",
                        type=str,
                        default="/h/mchoi/SpatioStuff/gifs")

    # Model params
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--hidden_channels", nargs="+")

    args = parser.parse_args()
    return args


def save(model, optim, args):
    """Save the model and optimizer states
    """
    torch.save({
        "convlstm": model.state_dict(),
        "optim": optim.state_dict(),
    }, args.checkpoint_path)


def load(args, model_args, device):
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


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.rdm_seed)

    if args.use_wandb is True:
        wandb.init(project="ConvLSTM on Moving MNIST", entity="mchoi")
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size
        }

    # Dataset
    transform = transforms.Compose([
        #transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_dset = MovingMnist(args.dset_path,
                             split="train",
                             transform=transform)
    val_dset = MovingMnist(args.dset_path, split="val", transform=transform)
    test_dset = MovingMnist(args.dset_path, split="test", transform=transform)

    train_dl = torch.utils.data.DataLoader(dataset=train_dset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
    val_dl = torch.utils.data.DataLoader(dataset=val_dset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_dset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=True)

    # Loss function
    loss_fn = nn.MSELoss()

    # Model and optim
    hidden_channels = tuple(map(int, args.hidden_channels))
    model_args = [args.in_channels, args.seq_len, hidden_channels]
    if os.path.isfile(
            args.checkpoint_path) and "None" not in args.checkpoint_path:
        model, optim = load(args, model_args, device)
    else:
        model = ConvLSTMForecastingNet(*model_args)
        model.to(device)  # Must put model on device before feeding into optim
        optim = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    
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

            # Sample tensor from last batch
            if "None" not in args.sample_output_path:
                sample_tensor = torch.cat((example[0], preds[0]), dim=0)
                tensor_to_gif(sample_tensor.detach().cpu(),
                              f"moving_mnist_val.gif", args.sample_output_path)

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

if __name__ == "__main__":
    args = prepare_args()
    main(args)
