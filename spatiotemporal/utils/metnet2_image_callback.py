import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torchvision
import wandb


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of a module.
    Notes:
        Images are stacked into a mosaic, with output on the top
        and input on the bottom
    """
    
    def __init__(self, val_samples):
        super().__init__()
        self.val_imgs, self.targets, self.leadtime_vecs = val_samples
          
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run inference on one sample at the start of each training epoch
        Args:
            trainer (Trainer): 
            pl_module (pl.Module):
        """
        if pl_module.global_rank == 0:  # Run only once across all nodes
            examples_batch = self.val_imgs.to(device=pl_module.device)
            targets_batch = self.targets.to(device=pl_module.device)
            leadtime_vecs_batch = self.leadtime_vecs.to(device=pl_module.device)

            pad = nn.ZeroPad2d((16, 16, 16, 16))
        
            print("**Creating sample**")
            samples = []
            for x, target, leadtime_vec in zip(examples_batch, targets_batch, leadtime_vecs_batch):
                x = x.unsqueeze(0)
                target = target.unsqueeze(0)
                pred = pl_module(x, leadtime_vec)
                ex, land = x.squeeze(0).chunk(2, dim=-3)
                sample = torch.cat((ex, pad(target), pad(pred), land))
                samples.append(sample)
            grid_images = torchvision.utils.make_grid(samples, nrow=6, value_range=(0, 1), normalize=True)
            #torchvision.utils.save_image(grid_images, "wandb_sample.png")

            caption = "Epoch sample"
            trainer.logger.experiment.log({
                "val/examples": [wandb.Image(grid_images, caption=caption)],
                "global_step": trainer.global_step
                })
