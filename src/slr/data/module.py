import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .pose import PoseInferenceDataset, PoseDataset
from .rgb_full import FullFrameInferenceDataset, FullFrameDataset


def get_inference_data_loader(language_id, **kwargs):
    """Get a data loader for predict.py for a given language ID.

    :param language_id: The language ID."""
    batch_size = kwargs['batch_size']
    data_kind = kwargs['data_kind']
    num_workers = kwargs['num_workers']
    data_dir = kwargs['data_dir']
    video_file_extension = kwargs["video_file_extension"]

    if data_kind.startswith('Mediapipe'):
        dataset = PoseInferenceDataset(data_dir, data_kind, video_file_extension, language_id, dict(kwargs))
    elif data_kind == 'RGB_Full':
        dataset = FullFrameInferenceDataset(data_dir, data_kind, '.mp4', language_id, dict(kwargs))
    else:
        raise ValueError(f'Unsupported data_kind {data_kind}.')

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      shuffle=True, collate_fn=get_collate_fn(data_kind, variable_length=True, inference_mode=True))


class DataModule(pl.LightningDataModule):
    def __init__(self, num_workers: int, batch_size: int, **kwargs):
        """Initialize the data module.

        :param num_workers: The number of workers for the data loaders.
        :param batch_size: Batch size."""
        super(DataModule, self).__init__()

        # Arguments.
        self.data_kind = kwargs['data_kind']
        self.num_workers = num_workers
        self.data_dir = kwargs['data_dir']
        self.batch_size = batch_size
        self.variable_length_sequences = kwargs['variable_length_sequences']
        self.samples_file_override = kwargs['samples_file_override']

        self.train_sets = []
        self.val_sets = []
        self.test_sets = []
        # Datasets per language.
        for lang_id, data_dir in enumerate(self.data_dir):
            # Initialization.
            if self.data_kind == 'Mediapipe':
                self.train_sets.append(PoseDataset('train', 'mediapipe', data_dir, lang_id, **kwargs))
                self.val_sets.append(PoseDataset('val', 'mediapipe', data_dir, lang_id, **kwargs))
                self.test_sets.append(PoseDataset('test', 'mediapipe', data_dir, lang_id, **kwargs))
            elif self.data_kind == 'RGB_Full':
                self.train_sets.append(FullFrameDataset(data_dir, 'train', lang_id, self.samples_file_override))
                self.val_sets.append(FullFrameDataset(data_dir, 'val', lang_id, self.samples_file_override))
                self.test_sets.append(FullFrameDataset(data_dir, 'test', lang_id, self.samples_file_override))
            else:
                raise ValueError(f'Unknown dataset kind {self.data_kind}.')

        # Final dataset: all languages.
        self.train_set = ConcatDataset(self.train_sets)
        self.val_set = ConcatDataset(self.val_sets)
        self.test_set = ConcatDataset(self.test_sets)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          shuffle=True, collate_fn=get_collate_fn(self.data_kind, self.variable_length_sequences))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=get_collate_fn(self.data_kind, self.variable_length_sequences))

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=get_collate_fn(self.data_kind, self.variable_length_sequences))

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument('--data_kind', type=str, help='The kind of data to load (e.g., RGB or keypoints).',
                            required=True)
        parser.add_argument('--num_workers', type=int, help='Number of DataLoader workers.', default=0)
        parser.add_argument('--data_dir', nargs='+', type=str, help='Root dataset directories, one per dataset.',
                            required=True)
        parser.add_argument('--variable_length_sequences', action='store_true', help='Use variable length sequences.')
        parser.add_argument('--samples_file_override', type=str, help='Name of the samples file (default: samples.csv)')
        parser.add_argument('--data_path_override', type=str,
                            help='Useful for trying out different post-processing steps. See pose.py for usage.',
                            default='mediapipe_post')
        parser.add_argument('--mediapipe_2d', action='store_true',
                            help='Add this flag to drop the third dimension for MediaPipe keypoints.')
        parser.add_argument('--source_aspect_ratio', nargs='+', type=float, required=True,
                            help='Aspect ratio of the source videos, one value per dataset.')
        # Augmentation flags for PTN.
        parser.add_argument('--augment_rotY', type=int,
                            help='If greater than zero, the maximum amount of degrees to rotate the pose with along the Y axis.',
                            default=0)
        parser.add_argument('--augment_hflip', action='store_true',
                            help='Enable random horizontal flipping with probability 0.5.')
        parser.add_argument('--augment_adaptSpeed', type=float,
                            help='Probability for adapting the speed augmentation. See also --augment_adaptSpeedFaster.',
                            default=0.0)
        parser.add_argument('--augment_adaptSpeedFaster', type=float,
                            help='Probability that adapting the speed will drop frames. If adapting the speed, but not dropping frames, will interpolate frames.',
                            default=0.5)
        parser.add_argument('--augment_temporalShift', type=float,
                            help='Probability for applying temporal shift augmentation.',
                            default=0.0)
        parser.add_argument('--learning_curve_pct', type=float, default=1.0)
        return parent_parser


def get_collate_fn(data_kind: str, variable_length: bool, inference_mode: bool = False):
    """Get a collate function to collate batches based on several settings.

    :param data_kind: The kind of data (poses or video).
    :param variable_length: Whether to return variable length sequences in batches (padded) or fixed length sequences.
    :param inference_mode: Set this to `True` in predict.py"""

    def collate_images(batch):
        clips = [e[0] for e in batch]
        targets = [e[1] for e in batch]
        filenames = [e[2] for e in batch]
        language_ids = [e[3] for e in batch]

        clips = torch.stack(clips).permute(1, 0, 2, 3, 4)  # B, T, C, H, W -> T, B, C, H, W.
        targets = torch.from_numpy(np.array(targets))
        language_ids = torch.tensor(language_ids).long()

        return clips, targets, filenames, language_ids

    def collate_poses_variable_length(batch):
        clips = [e[0] for e in batch]
        targets = [e[1] for e in batch]
        filenames = [e[2] for e in batch]
        language_ids = [e[3] for e in batch]

        clips = torch.nn.utils.rnn.pad_sequence(clips, batch_first=False, padding_value=float("nan"))
        targets = torch.from_numpy(np.array(targets))
        language_ids = torch.tensor(language_ids).long()

        return clips, targets, filenames, language_ids

    def collate_poses_inference(batch):
        clips = [e[0] for e in batch]
        targets = [0 for _ in batch]  # Dummy targets.
        filenames = [e[1] for e in batch]
        language_ids = [e[2] for e in batch]

        lengths = [len(clip) for clip in clips]

        clips = torch.nn.utils.rnn.pad_sequence(clips, batch_first=False)
        targets = torch.from_numpy(np.array(targets))
        language_ids = torch.tensor(language_ids).long()

        return clips, targets, filenames, language_ids, lengths

    def collate_poses(batch):
        clips = [e[0] for e in batch]
        targets = [e[1] for e in batch]
        filenames = [e[2] for e in batch]
        language_ids = [e[3] for e in batch]

        clips = torch.stack(clips).permute(1, 0, 2)  # B, T, C -> T, B, C.
        targets = torch.from_numpy(np.array(targets))
        language_ids = torch.tensor(language_ids).long()

        return clips, targets, filenames, language_ids

    if 'Mediapipe' in data_kind or 'OpenPose' in data_kind or 'MMPose' in data_kind:
        if inference_mode:
            return collate_poses_inference
        elif variable_length:
            return collate_poses_variable_length
        else:
            return collate_poses
    elif data_kind == 'RGB_Full':
        return collate_images
    else:
        raise ValueError(f'Unknown dataset kind {data_kind}.')
