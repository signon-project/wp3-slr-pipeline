"""Pose Transformer Network, implemented with PyTorch classes."""
from typing import List

import numpy as np
import torch
from torch import nn

from .common import KeypointEmbeddingDense, PositionalEncoding


def _normalize(keypoints: torch.Tensor) -> torch.Tensor:
    # Input shape: (frames, batch, keypoints, coordinates).
    # 1. Normalize the pose by shifting and scaling around the neck.
    lshoulder = keypoints[:, :, [11]]
    rshoulder = keypoints[:, :, [12]]
    neck = 0.5 * (lshoulder + rshoulder)
    dist = torch.linalg.norm(lshoulder - rshoulder, ord=2, dim=-1, keepdims=True)
    keypoints = (keypoints - neck) / dist

    # 2. Now do the same for both hands.
    lwrist = keypoints[:, :, [33]]
    lmiddlemcp = keypoints[:, :, [42]]
    dist = torch.linalg.norm(lmiddlemcp - lwrist, ord=2, dim=-1, keepdims=True)
    keypoints[:, :, 33:54] = (keypoints[:, :, 33:54] - lwrist) / dist

    rwrist = keypoints[:, :, [54]]
    rmiddlemcp = keypoints[:, :, [63]]
    dist = torch.linalg.norm(rmiddlemcp - rwrist, ord=2, dim=-1, keepdims=True)
    keypoints[:, :, 54:75] = (keypoints[:, :, 54:75] - rwrist) / dist

    return keypoints


class FeatureProcessing(nn.Module):
    def __init__(self):
        super(FeatureProcessing, self).__init__()

    def forward(self, pose_clips: torch.Tensor) -> torch.Tensor:
        # Input is a padded batch (pad value: NaN) containing multiple samples.
        # The shape is (length, batch, keypoints, coordinates).
        # Imputation and augmentation (if training) have already been applied.
        # In this module, we perform normalization and feature extraction.

        # Normalization: shift and scale.
        pose_clips = _normalize(pose_clips)
        # Selected keypoints: upper pose, left hand, right hand.
        keypoints = torch.cat([
            torch.arange(0, 25),
            torch.arange(33, 33 + 42),
        ])
        pose_clips = pose_clips[..., keypoints, :]
        # We only keep x and y, and drop z.
        pose_clips = pose_clips[..., :2]
        # If any features are needed other than x and y coordinates, here is where you'd extract them.
        # Or, if they need to be extracted based on 3D coordinates, then above the previous line.
        # Ravel.
        pose_clips = pose_clips.reshape(pose_clips.size(0), pose_clips.size(1), -1)
        return pose_clips


class PTN(nn.Module):
    def __init__(self, num_attention_layers: int, num_attention_heads: int, d_hidden: int, num_classes: List[int],
                 lang_embedding_size=4, **kwargs):
        super(PTN, self).__init__()

        # Hyperparameters / arguments.
        self.in_features = kwargs['d_pose']
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.d_hidden = d_hidden
        self.num_classes = num_classes
        self.residual_pose_embedding = kwargs['residual_pose_embedding']
        self.variable_length_sequences = kwargs['variable_length_sequences']
        self.use_pose_embedding = not kwargs['no_pose_embedding']
        self.pose_embedding_kind = kwargs['pose_embedding_kind']

        self.feature_extractor = FeatureProcessing()

        # Model architecture.
        if self.use_pose_embedding:
            if self.pose_embedding_kind == 'dense':
                self.pose_embedding = KeypointEmbeddingDense(self.in_features, self.d_hidden,
                                                             self.residual_pose_embedding)
            else:
                raise ValueError(f'Unknown pose embedding kind {self.pose_embedding_kind}.')
        else:
            self.pose_embedding = nn.Linear(self.in_features, self.d_hidden, bias=False)
        self.pos_enc = PositionalEncoding(self.d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(self.d_hidden, self.num_attention_heads, 2 * self.d_hidden,
                                                   dropout=0.2)
        self.self_attention = nn.TransformerEncoder(encoder_layer, self.num_attention_layers,
                                                    nn.LayerNorm(self.d_hidden))

        if len(num_classes) > 1:
            self.language_embedding = nn.Embedding(len(num_classes), lang_embedding_size)
        else:
            self.language_embedding = None

        self.classifier = nn.Linear(self.d_hidden + (0 if len(num_classes) == 1 else lang_embedding_size),
                                    int(np.sum(np.array(num_classes))))

        self.cls_emb = nn.Embedding(1, self.d_hidden)

    def forward(self, batch, language_ids) -> torch.Tensor:
        b = batch.size(1)
        pose_clip = self.feature_extractor(batch)

        mask = torch.isnan(pose_clip).all(dim=-1)  # True for padding.
        mask = torch.cat((torch.zeros(1, b, dtype=torch.bool, device=pose_clip.device), mask), dim=0)  # False for CLS.
        mask = mask.transpose(1, 0)

        pose_clip = torch.nan_to_num(pose_clip, 0)  # Set padding to zero.

        pose_embedded = self.pose_embedding(pose_clip)

        pose_embedded = self.pos_enc(pose_embedded)

        # Add CLS token.
        cls_inputs = torch.zeros((b, 1), dtype=torch.long).to(pose_embedded.device)
        cls_embedded = self.cls_emb(cls_inputs).permute(1, 0, 2)
        self_attention_inputs = torch.cat((cls_embedded, pose_embedded), dim=0)

        self_attention_outputs = self.self_attention(self_attention_inputs, src_key_padding_mask=mask)

        classifier_inputs = self_attention_outputs[0]  # CLS token output.

        if self.language_embedding is not None:
            emb_lang = self.language_embedding(language_ids)
            classifier_inputs = torch.cat((classifier_inputs, emb_lang), dim=-1)

        logits = self.classifier(classifier_inputs)

        return logits

    def init_output_bias(self, class_weights):
        self.classifier.bias.data = torch.log(class_weights)

    def setup_inference_hook(self, embedding_kind: str, hook):
        if embedding_kind == 'spatial':
            self.pose_embedding.register_forward_hook(hook)
        elif embedding_kind == 'temporal' or embedding_kind == 'CLS':
            self.self_attention.register_forward_hook(hook)
        else:
            raise ValueError(f'Unsupported embedding kind {embedding_kind} for inference hook.')
