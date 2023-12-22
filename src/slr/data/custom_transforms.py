"""Custom data transforms, e.g., for pose data."""
import math

import torch
from torch import nn


class RandomHorizontalFlip(nn.Module):
    """Randomly horizontally flip a pose with given probability.

    The face keypoints are not flipped because this is non-trivial due to the way MediaPipe's face mesh is constructed."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.horizontal_flip(x)
        return x

    @staticmethod
    def horizontal_flip(x):
        """
        Flip pose, left hand to right hand, and vice versa.
        face is not flipped.
        """
        x[..., 2] = 1.0  # Set to homogeneous coordinates (x, y, 1) by implicitly dropping z coordinate
        mirror_x_coordinate = (x[:, 11, 0] + x[:, 12, 0]) / 2
        shift_matrix = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).expand(x.shape[0], 3, 3).clone().to(
            x.device)
        shift_matrix[:, 0, 2] = mirror_x_coordinate
        flip_matrix = torch.tensor([[-1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).expand(x.shape[0], 3, 3).to(x.device)
        shift_back_matrix = (
            torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).expand(x.shape[0], 3, 3).clone().to(x.device)
        )
        shift_back_matrix[:, 0, 2] = -mirror_x_coordinate
        transform_matrix = torch.bmm(shift_matrix, torch.bmm(flip_matrix, shift_back_matrix))
        y = x.clone()
        # Flip right hand and put it on the left side
        y[:, 33:54, :] = torch.bmm(transform_matrix, x[:, 54:, :].permute(0, 2, 1)).permute(0, 2, 1)
        # Flip left hand and put it on the right side
        y[:, 54:, :] = torch.bmm(transform_matrix, x[:, 33:54, :].permute(0, 2, 1)).permute(0, 2, 1)
        # Flip pose (except face)
        y[:, 11:33:2, :] = torch.bmm(transform_matrix, x[:, 12:33:2, :].permute(0, 2, 1)).permute(0, 2, 1)
        y[:, 12:33:2, :] = torch.bmm(transform_matrix, x[:, 11:33:2, :].permute(0, 2, 1)).permute(0, 2, 1)
        return y


class Shift(nn.Module):
    """Add a random shift to the entire pose. The shift value is sampled from a normal distribution with the provided
    standard deviation."""

    def __init__(self, std=0.07):
        super().__init__()
        self.std = std

    def forward(self, x):
        shift = torch.randn(3).to(x.device) * self.std
        x = x + shift
        return x


class ShiftHandsIndividually(nn.Module):
    """With a certain probability, shift the hand keypoints with a value sampled from a normal distribution with given
    standard deviation. The hands are shifted with different values, but if one hand is shifted, both are."""

    def __init__(self, p=0.5, std=0.07):
        super().__init__()
        self.p = p
        self.std = std

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.shift(x)
        return x

    def shift(self, x):
        shift_left = torch.randn(3).to(x.device) * self.std
        shift_right = torch.randn(3).to(x.device) * self.std
        x[:, 33:54, :] += shift_left
        x[:, 54:75, :] += shift_right
        x[:, 15:22:2, :] = x[:, 15:22:2, :] + shift_left
        x[:, 16:23:2, :] = x[:, 16:23:2, :] + shift_right
        return x


class RotateHandsIndividually(nn.Module):
    """With a certain probability, rotate the hands individually with an angle up to a given value (in degrees).
    If one hand is rotated, both are."""

    def __init__(self, p=0.5, max_angle=15):
        super().__init__()
        self.p = p
        self.max_angle = max_angle

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.rotate_hands(x)
        return x

    def rotate_hands(self, x):
        angle_left = torch.rand(1).to(x.device) * self.max_angle * 2 - self.max_angle
        angle_right = torch.rand(1).to(x.device) * self.max_angle * 2 - self.max_angle
        x[:, 33:54, :] = self.rotate(x[:, 33:54, :], angle_left)
        x[:, 54:75, :] = self.rotate(x[:, 54:75, :], angle_right)
        x[:, 15:22:2, :] = self.rotate(x[:, 15:22:2, :], angle_left)
        x[:, 16:23:2, :] = self.rotate(x[:, 16:23:2, :], angle_right)
        return x

    @staticmethod
    def rotate(hand, angle):
        hand[..., 2] = 1.0  # Set to homogeneous coordinates (x, y, 1) by implicitly dropping z coordinate
        shift_matrix = (
            torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).expand(hand.shape[0], 3, 3).clone().to(hand.device)
        )
        shift_matrix[:, 0, 2] = hand[:, 0, 0]
        shift_matrix[:, 1, 2] = hand[:, 0, 1]
        shift_back_matrix = (
            torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).expand(hand.shape[0], 3, 3).clone().to(hand.device)
        )
        shift_back_matrix[:, 0, 2] = -hand[:, 0, 0]
        shift_back_matrix[:, 1, 2] = -hand[:, 0, 1]
        angle = angle * (math.pi / 180)
        rotation_matrix = (
            torch.tensor([[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]])
            .expand(hand.shape[0], 3, 3)
            .to(hand.device)
        )
        transform_matrix = torch.bmm(shift_matrix, torch.bmm(rotation_matrix, shift_back_matrix))
        return torch.bmm(transform_matrix, hand.permute(0, 2, 1)).permute(0, 2, 1)


class Zoom(nn.Module):
    """Randomly zoom in on the pose (i.e., rescale X, Y and Z) up to a certain degree."""

    def __init__(self, zoom=0.2):
        super().__init__()
        self.zoom = zoom

    def forward(self, x):
        zoom = 0.2
        rand_minval = 1.0 - zoom
        rand_maxval = 1.0 + zoom
        xscale = (rand_minval - rand_maxval) * torch.rand(1, 1, 1).to(x.device) + rand_maxval
        xscaled = xscale * x[..., 0:1]
        yscale = (rand_minval - rand_maxval) * torch.rand(1, 1, 1).to(x.device) + rand_maxval
        yscaled = yscale * x[..., 1:2]
        zscaled = x[..., 2:]
        x = torch.cat((xscaled, yscaled, zscaled), dim=-1)
        return x


class Jitter(nn.Module):
    """Introduce random Gaussian noise on the keypoints with a certain probability and standard deviation."""

    def __init__(self, p=0.5, hands_std=0.003, lips_std=0.001):
        super().__init__()
        self.p = p
        self.hands_std = hands_std
        self.lips_std = lips_std

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.jitter(x)
        return x

    def jitter(self, x):
        hands_jitter = torch.randn(x.shape).to(x.device) * self.hands_std
        x[:, 33:75, :] += hands_jitter[:, 33:75, :]
        # upper_lips_jitter = torch.randn(1).to(x.device) * self.lips_std
        # lower_lips_jitter = torch.randn(1).to(x.device) * self.lips_std
        # if (
        #         torch.mean(
        #             (
        #                     (x[:, lips_upper_inner_landmarks[5], 1] + upper_lips_jitter)
        #                     <= (x[:, lips_lower_inner_landmarks[5], 1] + lower_lips_jitter)
        #             ).float()
        #         )
        #         > 0.9
        # ):
        #     x[:, lips_upper_outer_landmarks, 1] = x[:, lips_upper_outer_landmarks, 1] + upper_lips_jitter
        #     x[:, lips_upper_inner_landmarks, 1] = x[:, lips_upper_inner_landmarks, 1] + upper_lips_jitter
        #     x[:, lips_lower_outer_landmarks, 1] = x[:, lips_lower_outer_landmarks, 1] + lower_lips_jitter
        #     x[:, lips_lower_inner_landmarks, 1] = x[:, lips_lower_inner_landmarks, 1] + lower_lips_jitter
        return x


class DropFrames(nn.Module):
    """With a given probability, drop a certain amount of frames from the sample."""

    def __init__(self, p=0.5, drop_ratio=0.1):
        super().__init__()
        self.p = p
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.drop_frames(x)
        return x

    def drop_frames(self, x):
        # Drop 10% of frames
        new_len = int(x.shape[0] * (1 - self.drop_ratio))
        if new_len == 0:
            return x
        unif = torch.ones(x.shape[0]).to(x.device)
        idx = unif.multinomial(new_len, replacement=False)
        idx, _ = torch.sort(idx)
        return x[idx]


class FrameHandDropout(nn.Module):
    """With a given probability, drop a certain amount of hand frames by setting them to NaN."""

    def __init__(self, p=0.5, drop_ratio=0.1):
        super(FrameHandDropout, self).__init__()
        self.p = p
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.drop_hand(x)
        return x

    def drop_hand(self, x):
        if torch.rand(1) < 0.5:  # Left hand versus right hand.
            hand_indices = torch.arange(33, 54)
        else:
            hand_indices = torch.arange(54, 75)
        # Select a random (drop_ratio * 100)% of the frames, and set the hand_indices of those frames to NaN.
        num_frames = int(math.floor(self.drop_ratio * len(x)))
        if num_frames > 0:
            frame_indices = torch.randint(0, len(x), (num_frames,))
            x[frame_indices][:, hand_indices] = float('nan')
        return x


class ShiftHandFrames(nn.Module):
    """Shift hand frames: move hands to different frames."""

    def __init__(self, p=0.5, max_frames_shift=10):
        super().__init__()
        self.p = p
        self.max_frames_shift = max_frames_shift

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.shift_hand_frames(x)
        return x

    def shift_hand_frames(self, x):
        # Shift max_frames_shift frames to the left or right
        max_frames_shift = min(self.max_frames_shift, max(len(x) // 10, 2))
        shift = torch.randint(1, max_frames_shift, (1,)).item()
        y = x.clone()
        if torch.rand(1) < 0.5:
            y = torch.cat([y[shift:], torch.full_like(y[:shift], float("nan"))])
        else:
            y = torch.cat([torch.full_like(y[:shift], float("nan")), y[:-shift]])

        # Only shift hands and pose
        # x[:, 468:, :] = y[:, 468:, :]
        # return x
        return y


class RandomCrop(nn.Module):
    """Select a random subfragment from the video."""

    def __init__(self, p=0.5, scale=(0.8, 1.0)):
        super().__init__()
        self.p = p
        self.scale = scale

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.random_crop(x)
        return x

    def random_crop(self, x):
        crop_length = torch.rand(1).to(x.device) * (self.scale[1] - self.scale[0]) + self.scale[0]
        crop_length = int(crop_length * x.shape[0])
        if crop_length == 0:
            return x
        crop_shift = torch.randint(0, max(x.shape[0] - crop_length, 1), (1,)).item()
        return x[crop_shift: crop_shift + crop_length]


class Passthrough:
    def __init__(self):
        pass

    def __call__(self, poses):
        return poses
