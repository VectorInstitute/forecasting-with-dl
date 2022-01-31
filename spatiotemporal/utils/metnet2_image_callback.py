import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""
    
    def __init__(self, val_samples, max_samples=4):
        super().__init__()
        self.val_imgs, _ = val_samples
        self.val_imgs = self.val_imgs[:max_samples]
          
    def on_validation_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
    
    
        leadtime_vecs = nn.functional.one_hot(torch.arange(10).long(), num_classes=10)
        preds = []
        for sample in self.val_images:
            for leadtime_vec in leadtime_vecs:
                preds.append(pl_modul(x, leadtime_vec))
        preds = torch.Tensor(preds)
        sample_tensor = torch.cat((example, target, preds), dim=0)


        mosaics = torch.cat([outs, val_imgs], dim=-2)
        caption = "Top: Output, Bottom: Input"
        trainer.logger.experiment.log({
            "val/examples": [wandb.Image(mosaic, caption=caption) 
                              for mosaic in mosaics],
            "global_step": trainer.global_step
            })
            
...

trainer = pl.Trainer(
    ...
    callbacks=[WandbImageCallback(val_samples)]
)
