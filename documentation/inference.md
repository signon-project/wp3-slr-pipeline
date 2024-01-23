# Inference documentation

*version 0.6, 2023/10/16*

Previous versions:

- *version 0.1, 2022/10/28*
- *version 0.2, 2023/01/23*
- *version 0.3, 2023/01/23*
- *version 0.4, 2023/05/12*
- *version 0.5, 2023/05/22*
- *version 1.0, 2023/12/22*

This file describes the way this codebase can be used to extract embeddings from SLR models.

## Code architecture

### Entry point

The entry point for inference is `src/predict.py`. This script has significantly fewer command line arguments
(CLAs) than `src/train.py`, because most of the information can be inferred from the checkpoint.

This script runs on a directory of video files, and writes embeddings to another directory.

### Data loading

The `InferenceDataset` performs keypoint extraction and processing on the video data.
It creates variable length sequences.

### Extracting embeddings

Each model has a `setup_inference_hook` function, which registers a hook on the layer
which outputs represent the extracted embedding. This hook is implemented in `predict.py`.
It stores the model's intermediate outputs and these can then be extracted and written to files.

The following output types are currently supported (they can be set with the `--embedding_kind` CLA).

- `"spatial"`: Per-frame pose embeddings. Will only contain spatial information.
- `"temporal"`: Per-frame embeddings after self-attention. Will contain temporal information.
- `"CLS"`: The CLS token embedding. Summarizes the video into one vector.
- `"probabilities"` The gloss probabilities. Summarizes the video into one vector.

## SLR models

The following SLR models are available.

| Language | Model                    | Description          | Validation accuracy |
|----------|--------------------------|----------------------|---------------------|
| NGT      | 3e-4 4x8 192-PoseFormer  | Trained from scratch | 47.26%              |
| VGT      | 3e-4 4x8 192-PoseFormer  | Transfer from NGT    | 52.37%              |
| ISL      | 3e-4 4x8 192-PoseFormer  | Transfer from NGT    | 30.10%              |
| BSL      | 3e-4 4x8 192-PoseFormer  | Transfer from NGT    | 32.78%              |
| LSE      | 3e-4 2x8 192-PoseFormer  | Transfer from NGT    | 59.40%              |

These models output `192`-dimensional embeddings.

These SLR models can be downloaded from the [SignON HuggingFace page](https://huggingface.co/signon-project).

### Previous versions

#### v0.5

| Language | Model                     | Description          | Validation accuracy | Download link                                            | Commit  |
|----------|---------------------------|----------------------|---------------------|----------------------------------------------------------|---------|
| NGT      | 3e-4 4x8 192-PoseFormer   | Trained from scratch | 47.26%              | https://cloud.ilabt.imec.be/index.php/s/c4wPQJYpXZNo9zp  | fa296f4 |
| VGT      | 3e-4 4x8 192-PoseFormer   | Transfer from NGT    | 52.37%              | https://cloud.ilabt.imec.be/index.php/s/aqS8L4yKAstyCPL  | fa296f4 |
| ISL      | 3e-4 4x8 192-PoseFormer   | Transfer from NGT    | 30.10%              | https://cloud.ilabt.imec.be/index.php/s/7XQzTCwwnJ9wNJC  | fa296f4 |
| BSL      | 3e-4 4x8 192-PoseFormer   | Transfer from NGT    | 32.78%              | https://cloud.ilabt.imec.be/index.php/s/gqrXMTJFKSR5PtS  | fa296f4 |

These models output `192`-dimensional embeddings.

#### v0.3

| Language | Model | Description                                                     | Validation accuracy | Download link                                           | Commit  |
|----------|-------|-----------------------------------------------------------------|---------------------|---------------------------------------------------------|---------|
| NGT      | PTN   | Pose Transformer Network (variable length) trained from scratch | 44.19%              | https://cloud.ilabt.imec.be/index.php/s/sFTjmMjGaz69Kp2 | hotfix |
| VGT      | PTN   | Pose Transformer Network (variable length) trained from scratch | 48.15%              | https://cloud.ilabt.imec.be/index.php/s/GKiHJRDRk7ajr5d | hotfix |
| ISL      | PTN   | Pose Transformer Network (variable length) transfer from NGT    | 11.5%                   | https://cloud.ilabt.imec.be/index.php/s/L8TWncKPk3drPro | hotfix |
| BSL      | PTN   | Pose Transformer Network (variable length) transfer from NGT    | 20%                   | https://cloud.ilabt.imec.be/index.php/s/pBJ4wRZHqLMp8CR | hotfix |

### Example

The following command would extract embeddings from all videos in `/tmp/videos` and write them to `/tmp/embeddings`,
using the VGT-292 checkpoint that is also stored in `/tmp`.

```shell
python -m predict /tmp/ptnd_vgt292_599feb0.ckpt /tmp/videos /tmp/embeddings \
  --batch_size 64 --embedding_kind spatial
```

For this example, every file in `/tmp/embeddings` will be a `.npy` file, containing an array of shape
`(N, 128)` where `N` is the number of frames in each video.

### Troubleshooting

- Getting CUDA out of memory errors? Entire sequences are loaded into memory. Consider lowering the batch size.
