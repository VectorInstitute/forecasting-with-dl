import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchvision
import wandb


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""
    
    def __init__(self, val_samples):
        super().__init__()
        self.val_imgs, self.targets, self.leadtime_vecs = val_samples
          
    def on_train_epoch_start(self, trainer, pl_module):
        examples_batch = self.val_imgs.to(device=pl_module.device)
        targets_batch = self.targets.to(device=pl_module.device)
        leadtime_vecs_batch = self.leadtime_vecs.to(device=pl_module.device)
    
        print("**Creating sample**")
        samples = []
        for x, target, leadtime_vec in zip(examples_batch, targets_batch, leadtime_vecs_batch):
            x = x.unsqueeze(0)
            target = target.unsqueeze(0)
            leadtime_vec = leadtime_vec.unsqueeze(0)    # Manually add batch dimensions
            pred = pl_module(x, leadtime_vec)
            sample = torch.cat((x.squeeze(0), target, pred), dim=0)
            samples.append(sample)
        grid_images = torchvision.utils.make_grid(samples, nrow=6)
        #torchvision.utils.save_image(grid_images, "wandb_sample.png")

        caption = "Epoch sample"
        trainer.logger.experiment.log({
            "val/examples": [wandb.Image(grid_images, caption=caption)],
            "global_step": trainer.global_step
            })
