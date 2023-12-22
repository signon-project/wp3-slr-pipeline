# Training documentation

*version 0.1, 2022/10/28*

This file describes the way this codebase can be used to train SLR models.

## Code architecture

### Entry point

The entry point for training is `src/train.py`. You can invoke this script with command line arguments (CLAs) that will
determine aspects of the training run such as the data location, feature type, model type, and model architecture.

This script creates a `DataModule` (see `src/slr/data/module.py`) and `Module` (see `src/slr/model/module.py`).
The `DataModule` creates `DataSet`s depending on the passed `--data_kind` CLA. Every dataset may have its own
CLAs, which can be found at the bottom of the `data/module.py` script.
The `Module` creates the required model architecture and handles training. The model architecture depends
on the `--model_name` CLA. Here as well, all CLAs can be found at the bottom of the file.

The training run is performed using `PyTorch Lightning`. Logs are written to the log directory passed with `--log_dir`.

Training progress is logged using WandB by default with a Tensorboard fallback. Set the `WANDB_API_KEY` to use WandB.

### Data preparation

To speed up training, the datasets are already preprocessed (unlike for inference, where we work directly on videos).
The preprocessing code to extract and process keypoints is found in `src/slr/data_preprocessing`.
After extracting the `.mp4` files of your dataset, along with the `samples.csv` file, you can run this code
to extract and process keypoints. The functions of interest are `run_mediapipe` and `postprocess`.

During inference, the exact same code will be used to extract and process keypoints from videos, which will make sure
that the input is in the correct format.

### Data loading

Samples are loaded using dataset classes, implemented in `src/slr/data`.
The dataset classes load individual sign instances: entire videos which correspond to a single gloss.
The `InferenceDataset` is a special case, which is discussed in `inference.md`.

The data loaders, which are created by the `DataModule`, use a custom collate function that returns subclasses
of `Batch` (also defined in `src/slr/data/module.py`). There are three kinds of batches:
`FixedLengthBatch`, `VariableLengthBatch`, and `InferenceBatch`. The first works with sequences of fixed length,
the second will use zero padding to allow for variable length sequences. The third is only used for inference.
Variable length sequences can be enabled using the `--variable_length_sequences` CLA.

### Transfer learning

Transfer learning is supported: you can pass a checkpoint using `--checkpoint`. You can choose to freeze
certain layers using `--freeze_parts`, which accepts a single layer name or comma separated list.
These layers will remain frozen, unless you also use `--freeze_until_convergence`, which will thaw the layers
when the validation loss has converged. Finally, you can set a new learning rate after they are thawed, using
`--lr_after_unfreeze`.
