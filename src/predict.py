import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch

from slr.data.module import get_inference_data_loader
from slr.models.module import Module
from train import _get_git_commit_hash


class OutputWriter:
    def __init__(self, output_directory: str):
        """Initialize the output writer.

        :param output_directory: Outputs will be written to this directory."""
        self.output_directory = output_directory

    def write(self, basename_original: str, data: np.ndarray):
        """Write an output file.

        :param basename_original: Basename of the original file.
        :param data: Embedding outputs."""
        filename, _extension = os.path.splitext(basename_original)
        output_file = os.path.join(self.output_directory, filename + '.npy')
        np.save(output_file, data)


def predict(args):
    """Output sign language representations for video data.

    :param args: The command line arguments."""
    # --- Initialization --- #
    module = Module.load_from_checkpoint(args.checkpoint)
    hparams = dict(module.hparams)

    pl.seed_everything(hparams['seed'])

    # Override the data directory.
    hparams['data_dir'] = args.data
    # Override the batch size.
    hparams['batch_size'] = args.batch_size
    # Override the number of workers
    hparams['num_workers'] = 1  # OpenCV does not play nice with >1 worker.
    # We don't need weighted loss for inference.
    hparams['weighted_loss'] = False
    # We don't want bias initialization for inference.
    hparams['use_bias_init'] = False
    hparams['language_id'] = args.language_id

    hparams["video_file_extension"] = args.video_file_extension

    output_writer = OutputWriter(args.output_directory)

    data_loader = get_inference_data_loader(**hparams)

    module = module.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        module = module.cuda()

    hook_outputs = []

    def _hook_spatial(self, input, output):
        hook_outputs.append(output)

    def _hook_temporal(self, input, output):
        hook_outputs.append(output[:, 1:])  # CLS is the first element, we don't want it.

    def _hook_cls(self, input, output):
        hook_outputs.append(output[:, 0])  # CLS is the first element.

    if args.embedding_kind == "spatial":
        module.model.setup_inference_hook(args.embedding_kind, _hook_spatial)
    elif args.embedding_kind == "temporal":
        module.model.setup_inference_hook(args.embedding_kind, _hook_temporal)
    elif args.embedding_kind == "CLS":
        module.model.setup_inference_hook(args.embedding_kind, _hook_cls)
    else:
        if args.embedding_kind != "probabilities":
            raise ValueError(f"Unknown embedding kind {args.embedding_kind}")

    # --- Logging --- #
    git_commit_hash = _get_git_commit_hash()
    if git_commit_hash not in args.checkpoint:
        print(f'[WARNING] The commit hash {git_commit_hash} was not found in the checkpoint path {args.checkpoint}. '
              f'Are you sure you are running the correct version of this repository?')

    # --- Inference --- #
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            model_inputs, _targets, filenames, language_ids, lengths = batch
            if use_cuda:
                model_inputs = model_inputs.to('cuda')
                _targets = _targets.to('cuda')

            model_outputs = module(model_inputs, language_ids)

            for sample_index, filename in enumerate(filenames):
                basename = filename.split('/')[-1]  # Cannot use os.path.basename in case the name is a directory.
                if args.embedding_kind != "probabilities":
                    if args.embedding_kind == "CLS":
                        output = hook_outputs[0][sample_index]
                    else:
                        output = hook_outputs[0][sample_index]
                        output = output[0:lengths[sample_index]]  # Unpad.
                    output_writer.write(basename, output.detach().cpu().numpy())
                else:
                    output = model_outputs[sample_index]
                    output_writer.write(basename, output.detach().cpu().numpy())

            # Clear hook storage.
            hook_outputs = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, help='The path to the checkpoint file.')
    parser.add_argument('data', type=str, help='The path to the directory containing the videos.')
    parser.add_argument('output_directory', type=str, help='The path to the directory where features will be written.')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing.', default=64)
    parser.add_argument('--embedding_kind', type=str,
                        help='Which embedding kind to output. One of ["spatial", "temporal", "CLS", "probabilities"]',
                        required=True)
    parser.add_argument('--video_file_extension', type=str, help='File extension for videos to be processed.',
                        default="mp4")
    parser.add_argument('--language_id', type=int, help='The language ID that we are testing for.', default=0)
    args = parser.parse_args()
    predict(args)
