# SLR Pipeline

This repository contains the code for training of and inference with SLR models.

The main entry points are `slr/train.py`, `slr/test.py` and `slr/predict.py`.
You should only use `slr/predict.py` unless you are developing new SLR models.

## Requirements

This Python code requires the following packages:

- PyTorch
- PyTorch Lightning
- Torchvision
- Scikit-learn
- NumPy
- MediaPipe
- OpenCV
- Tensorboard
- Torchmetrics
- Wandb

The exact versions can be found in `requirements.txt`.

These dependencies can be installed by creating a new virtual environment and running

```shell
pip install -r requirements.txt
```

## Usage

### SLR model development

This code base uses PyTorch Lightning to develop new models. The hyperparameters can be configured through command line
arguments. According to PyTorch Lightning guidelines, these are added in the `Module` and `DataModule` classes.

These classes delegate to the actual models and datasets. To add a new model, you should therefore implement the
model and delegate in the `Module` towards it. The same is the case for adding new dataset kinds.

### SLR model inference

For inference, we have two available modes: online and offline inference.
Online inference is for the purpose of the app. A video comes in, and SL representations (sequences of vectors) come
out.
This is supposed to happen in real-time.
Offline inference is for the purpose of WP3-WP4 interaction. A directory with videos is given to the inference module,
and the SL representations are written to another directory. Then, WP4 can use the resulting embeddings to train
SLT models.

#### Online inference

Please refer to
the [slr-component](https://github.com/signon-project/wp3-slr-component/tree/main/web_service/feature_extractor/slr_pipeline)
repository for online inference.

#### Offline inference.

To perform offline inference, you need to pick one of the checkpoints (see `documentation/inference.md`), download it, and run

```shell
python predict.py "${CHECKPOINT_PATH}" "${DATASET}" "${OUT}"
```

where `$DATASET` is the path to the directory containing videos, and ${OUT} is an empty directory to which the embedding
vectors will be written. For more details, see `documentation/inference.md`.

The checkpoint itself contains all the necessary information to determine which data to load and how to process them.

#### Available checkpoints

See `documentation/inference.md`.

# LICENSE

This code is licensed under the Apache License, Version 2.0 (LICENSE or http://www.apache.org/licenses/LICENSE-2.0).

