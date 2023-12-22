import argparse
import datetime
import os
from subprocess import CalledProcessError
from typing import List

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from slr.data.module import DataModule
from slr.models.module import Module
from slr.utils.callbacks import FinetuningCallback


def _get_callbacks(args, log_path) -> List[Callback]:
    """Get PyTorch Lightning callbacks.

    :param args: Command line arguments.
    :log_path: The log directory path.
    :return: A list of callbacks."""
    callbacks = []

    if not args.no_early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=20, mode='min'))
        callbacks.append(
            ModelCheckpoint(dirpath=os.path.join(args.log_dir, log_path) + '/checkpoints', mode='max',
                            filename='best_{epoch:02d}-{val_accuracy:.2f}', monitor='val_accuracy', save_top_k=1))
    else:
        # Save the last checkpoint.
        callbacks.append(
            ModelCheckpoint(dirpath=os.path.join(args.log_dir, log_path) + '/checkpoints', filename='last_{epoch:02d}',
                            save_last=True))

    if args.freeze_until_convergence:
        callbacks.append(FinetuningCallback(args.lr_after_unfreeze, args.keep_frozen))

    callbacks.append(LearningRateMonitor())

    callbacks.append(ModelSummary(5))

    return callbacks


def _get_git_commit_hash() -> str:
    """Get the hash of the current git commit.

    :return: The short hash string."""
    import subprocess
    try:
        hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print(f'Git commit hash is {hash}.')
        return hash
    except CalledProcessError:
        print('Failed to get git commit hash. We will write logs to the root of the log directory!')
        return ""


def train(args):
    """Start the training process.

    :param args: The command line arguments."""
    # --- Initialization --- #
    pl.seed_everything(args.seed)

    module = Module(**vars(args))
    data_module = DataModule(**vars(args))

    git_commit_hash = _get_git_commit_hash()
    name = args.run_name if args.run_name is not None else args.model_name
    log_dir_name = name + '_' + git_commit_hash + '_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    callbacks = _get_callbacks(args, log_dir_name)

    # --- Loading checkpoint --- #
    if args.checkpoint is not None:
        module.load_weights(args.checkpoint)

    if args.freeze_parts is not None:
        module.freeze_part(args.freeze_parts)

    # --- Logging --- #
    if 'WANDB_API_KEY' in os.environ.keys():
        logger = WandbLogger(log_dir_name, args.log_dir, project='slr', log_model=True)
    else:
        print('Environment variable `WANDB_API_KEY` is not set, using Tensorboard instead.')
        log_dir = os.path.join(args.log_dir, git_commit_hash)
        logger = TensorBoardLogger(log_dir)

    # --- Training --- #
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    trainer.fit(module, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--retrain_on_all', action='store_true', help='Train on training and validation set data.')
    parser.add_argument('--lr_steps', nargs='+', type=int,
                        help='Used only if --retrain_on_all is passed; sets the step indices to drop the learning rate.')
    parser.add_argument('--run_name', type=str, help='Name for the run in WANDB logs', default=None)
    parser.add_argument('--log_dir', type=str, help='Path to the log directory.', required=True)
    parser.add_argument('--no_early_stopping', action='store_true', help='Disable early stopping.')
    parser.add_argument('--freeze_parts', type=str, help='Comma separated list of parts to freeze.')
    parser.add_argument('--freeze_until_convergence', action='store_true',
                        help='Freeze transferred weights until the validation loss has stopped improving, then unfreeze them.')
    parser.add_argument('--keep_frozen', type=str,
                        help='Comma separated list of parts to keep frozen even after convergence.')
    parser.add_argument('--lr_after_unfreeze', type=float, help='Learning rate for fine-tuning after unfreezing.')
    parser.add_argument('--seed', type=int, help='Random seed.', default=42)
    parser.add_argument('--checkpoint', type=str, help='Load weights from a pre-trained checkpoint')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Module.add_model_specific_args(parser)
    parser = DataModule.add_datamodule_specific_args(parser)

    args = parser.parse_args()

    train(args)
