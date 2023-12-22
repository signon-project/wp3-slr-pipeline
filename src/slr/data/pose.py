import glob
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from . import custom_transforms as CT
from .common import Sample, collect_samples
from ..data_preprocessing.mediapipe_keypoints import extract, _temporal_imputation, _rescale_joints


def get_augmentation_transforms(hparams: Dict, *, job):
    """Get the transforms required to perform augmentation. For validation and testing, these do nothing.

    :param hparams: Module hyperparameters.
    :param job: The job."""
    if not job == 'train':
        return CT.Passthrough()
    if hparams['data_kind'].startswith('Mediapipe'):
        return T.Compose([
            CT.ShiftHandFrames(p=0.2),
            CT.RandomHorizontalFlip(p=0.2),
            CT.RotateHandsIndividually(p=0.2),
            CT.Jitter(p=0.2),
            CT.RandomCrop(p=0.2),
            CT.DropFrames(p=0.2, drop_ratio=0.1),
            CT.FrameHandDropout(p=0.2, drop_ratio=0.1)
        ])
    else:
        raise ValueError('No transforms implemented for data kind {}'.format(hparams['data_kind']))


class PoseInferenceDataset(Dataset):
    """Dataset for inference mode. Extracts keypoints from image data."""

    def __init__(self, directory: str, data_kind: str, file_extension: str, language_id: int, hparams: Dict):
        """Create the PoseInferenceDataset.

        :param directory: Path to the directory containing videos or image directories.
        :param data_kind: The kind of data that will be passed on to the model (MediaPipe / RGB).
        :param file_extension: File extension of the data.
        :param language_id: ID of the language we are doing inference for.
        :param hparams: Module hyperparameters."""
        super(PoseInferenceDataset, self).__init__()

        if file_extension[0] == '.':
            file_extension = file_extension[1:]
        self.file_extension = file_extension
        self.aspect_ratio = hparams['source_aspect_ratio']

        print(f'Inference dataset will load files from {directory}.')

        self.videos = True
        if 'jpg' in file_extension or 'png' in file_extension:
            print(f'File extension is an image extension ({file_extension}). Will load image directories.')
            self.videos = False

        self.path = directory
        self.data_kind = data_kind
        if self.videos:
            self.files = sorted(
                glob.glob(os.path.join(self.path, '**', f'*.{file_extension}'), recursive=True))  # Video files.
        else:
            self.files = sorted(glob.glob(os.path.join(self.path, '**/*/')))  # Image directories.
        self.sharpen_flag = hparams.get("sharpen", False)
        self.sharpen_sigma = hparams.get("sharpen_sigma", 0)
        self.debug = hparams.get("debug", False)
        self.language_id = language_id

    def __getitem__(self, item):
        """Get the input features and filename as a tuple."""
        frames = []
        if self.videos:
            # Load video.
            cap = cv2.VideoCapture(self.files[item])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            # Load image directory.
            for file in sorted(glob.glob(os.path.join(self.files[item], f'*.{self.file_extension}'))):
                frame = cv2.imread(file)
                if frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        height, width, _ = frames[0].shape

        # Perform any processing on the video, like extracting keypoints.
        if self.data_kind.startswith('Mediapipe'):
            keypoints = self._extract_keypoints(np.stack(frames), self.sharpen_flag, self.sharpen_sigma)

            keypoints = _rescale_joints(keypoints, self.aspect_ratio)
            keypoints = _temporal_imputation(keypoints)  # Perform temporal imputation (interpolation/extrapolation).
            keypoints = np.nan_to_num(keypoints)  # Clean up any remaining NaNs (= constant imputation).

            keypoints = torch.from_numpy(keypoints).float()
            return keypoints, self.files[item], self.language_id
        else:
            raise ValueError(f'Unsupported data kind {self.data_kind}')

    def _extract_keypoints(self, frames: np.ndarray, sharpen: bool, sharpen_sigma: float) -> np.ndarray:
        if sharpen:
            return extract(frames, sharpen_fn=sharpen, sharpen_sigma=sharpen_sigma)
        else:
            return extract(frames)

    def __len__(self):
        return len(self.files)


class PoseDataset(Dataset):
    """Used during training and testing: loads keypoints."""

    def __init__(self, job: str, pose_format: str, root_path: str, lang_id: int, **kwargs):
        """Create a PoseDataset for training or testing.

        :param job: "train", "val" or "test".
        :param pose_format: The pose format, only here for backwards compatibility.
        :param root_path: The root data directory.
        :param lang_id: Which language ID these data belong to."""
        super(PoseDataset, self).__init__()

        self.root_path = root_path
        self.job = job
        self.samples_file_override = kwargs['samples_file_override']
        self.variable_length = kwargs['variable_length_sequences']
        self.pose_format = pose_format
        if type(kwargs['source_aspect_ratio']) == float:
            self.aspect_ratio = kwargs['source_aspect_ratio']
        else:
            self.aspect_ratio = kwargs['source_aspect_ratio'][lang_id]
        self.lang_id = lang_id
        self.retrain_on_all = kwargs['retrain_on_all']
        self.learning_curve_pct = kwargs['learning_curve_pct']

        self.augment = get_augmentation_transforms(dict(**kwargs), job=self.job)

        self.samples = self._collect_samples()

    def __getitem__(self, item):
        sample = self.samples[item]
        sample_filename, _ = os.path.splitext(sample.path)
        clip = np.load(os.path.join(self.root_path, self.pose_format, sample_filename + '.npy'))

        clip = _rescale_joints(clip, self.aspect_ratio)
        clip = _temporal_imputation(clip)  # Perform temporal imputation (interpolation/extrapolation).
        clip = np.nan_to_num(clip)  # Clean up any remaining NaNs (= constant imputation).

        clip = torch.from_numpy(clip).float()

        clip = self.augment(clip)

        return clip, sample.label, sample.path, self.lang_id

    def __len__(self):
        return int(self.learning_curve_pct * len(self.samples))

    def _collect_samples(self) -> List[Sample]:
        return collect_samples(self.root_path, self.job, self.samples_file_override, self.retrain_on_all)
