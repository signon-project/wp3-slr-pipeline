"""PyTorch Lightning Callbacks."""
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class FinetuningCallback(Callback):
    """Callback that enables unfreezing layers after the validation loss has stopped improving.
    Useful for fine-tuning pre-trained models."""

    def __init__(self, lr_after_unfreeze: Optional[float], keep_frozen: Optional[str]):
        """Initialize the FinetuningCallback.

        :param lr_after_unfreeze: The learning rate will be set to this value for all model parameters after unfreezing.
        :param keep_frozen: A string containing a comma separated list of names of layers to keep frozen after convergence."""
        super(FinetuningCallback, self).__init__()

        self.enabled = True
        self.patience = 5
        self.epochs_since_last_improvement = 0
        self.previous_best_value = torch.inf
        self.lr_after_unfreeze = lr_after_unfreeze
        self.keep_frozen = keep_frozen

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return

        logs = trainer.callback_metrics

        current_loss = logs["val_loss"].squeeze()
        if current_loss < self.previous_best_value:
            self.previous_best_value = current_loss
            self.epochs_since_last_improvement = 0
        else:
            self.epochs_since_last_improvement += 1

        if self.epochs_since_last_improvement >= self.patience:
            print(f'{self.epochs_since_last_improvement} epochs without improvement. Will now thaw the frozen layers.')
            pl_module.unfreeze()
            if self.keep_frozen is not None:
                pl_module.freeze_part(self.keep_frozen)
            if self.lr_after_unfreeze is not None:
                print(f'Also, we will update the learning rate to {self.lr_after_unfreeze} for fine-tuning.')
                optimizers = pl_module.optimizers(False)
                if not isinstance(optimizers, list):
                    optimizers = [optimizers]
                for optimizer in optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr_after_unfreeze
            self.enabled = False
