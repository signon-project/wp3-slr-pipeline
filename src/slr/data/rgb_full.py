"""Dataset that loads full frame RGB images."""
import glob
import os
from typing import List, Optional, Dict

import cv2
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

from .common import collect_samples, Sample

IMAGE_RESIZE_SIZE = 256
IMAGE_CROP_SIZE = 224
NORM_MEAN_IMGNET = [0.485, 0.456, 0.406]
NORM_STD_IMGNET = [0.229, 0.224, 0.225]


def get_transforms(hparams: Dict, *, train: bool):
    if train:
        return T.Compose([
            T.Resize(IMAGE_RESIZE_SIZE),
            T.RandomResizedCrop(IMAGE_CROP_SIZE),
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET)  # TODO: Replace with dataset specific values.
        ])
    else:
        return T.Compose([
            T.Resize(IMAGE_RESIZE_SIZE),
            T.CenterCrop(IMAGE_CROP_SIZE),
            T.ConvertImageDtype(torch.float),
            T.Normalize(NORM_MEAN_IMGNET, NORM_STD_IMGNET)  # TODO: Replace with dataset specific values.
        ])


class FullFrameInferenceDataset(Dataset):
    def __init__(self, directory: str, data_kind: str, file_extension: str, language_id: int, hparams: Dict):
        super(FullFrameInferenceDataset, self).__init__()

        if file_extension[0] == '.':
            file_extension = file_extension[1:]

        print(f'Inference dataset will load files from {directory}.')

        self.path = directory
        self.data_kind = data_kind
        self.files = sorted(glob.glob(os.path.join(self.path, f'*.{file_extension}')))
        self.transforms = get_transforms(hparams, train=False)
        self.language_id = language_id

    def __getitem__(self, item):
        """Get the input features and filename as a tuple."""
        # Load video.
        frames = []
        cap = cv2.VideoCapture(self.files[item])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        clip = self.transforms(frames)

        return clip, self.files[item], self.language_id

    def __len__(self):
        return len(self.files)


class FullFrameDataset(Dataset):
    def __init__(self, root_path: str, job: str, lang_id: int, samples_file_override: Optional[str]):
        super(FullFrameDataset, self).__init__()

        self.root_path = root_path
        self.job = job
        self.samples_file_override = samples_file_override
        self.lang_id = lang_id

        self.transform = get_transforms({'data_kind': 'RGB_Full'}, train=self.job == 'train')

        self.samples = self._collect_samples()

    def __getitem__(self, item):
        sample = self.samples[item]
        frames, _, _ = torchvision.io.read_video(os.path.join(self.root_path, 'clips', sample.path), pts_unit='sec',
                                                 output_format='TCHW')

        if sample.frame_indices is None:
            clip = frames[self._get_frame_indices(len(frames))]
        else:
            clip = frames[sample.frame_indices]

        clip = self.transform(clip)

        return clip, sample.label, sample.path, self.lang_id

    def _get_frame_indices(self, num_frames: int) -> List[int]:
        # TODO: Make these values configurable.
        sequence_length = 16
        temporal_stride = 2
        frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
        frame_end = frame_start + sequence_length * temporal_stride
        if frame_start < 0:
            frame_start = 0
        if frame_end > num_frames:
            frame_end = num_frames
        frame_indices = list(range(frame_start, frame_end, temporal_stride))
        while len(frame_indices) < sequence_length:
            # Pad
            frame_indices.append(frame_indices[-1])
        return frame_indices

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self) -> List[Sample]:
        return collect_samples(self.root_path, self.job, self.samples_file_override)
