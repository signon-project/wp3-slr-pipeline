"""Video Transformer Network, implemented with PyTorch classes."""
from typing import List

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights, efficientnet_b0, \
    EfficientNet_B0_Weights

from .common import PositionalEncoding


class VTN(nn.Module):
    def __init__(self, backbone_name: str, num_attention_layers: int, num_attention_heads: int, d_hidden: int,
                 num_classes: List[int], lang_embedding_size=4):
        super(VTN, self).__init__()

        # Hyperparameters / arguments.
        self.backbone_name = backbone_name
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.d_hidden = d_hidden
        self.num_classes = num_classes

        # Model architecture.
        if self.backbone_name == 'resnet34':
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2],
                                          nn.Conv2d(512, self.d_hidden, 1) if self.d_hidden != 512 else nn.Identity(),
                                          nn.AdaptiveAvgPool2d((1, 1)))
        elif self.backbone_name == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2],
                                          nn.Conv2d(2048, self.d_hidden, 1) if self.d_hidden != 2048 else nn.Identity(),
                                          nn.AdaptiveAvgPool2d((1, 1)))
        elif self.backbone_name == 'efficientnet_b0':
            efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights)
            efficientnet.classifier = nn.Identity()
            self.backbone = nn.Sequential(efficientnet,
                                          nn.Linear(1280, self.d_hidden) if self.d_hidden != 1280 else nn.Identity())
        else:
            raise ValueError(f'Unsupported backbone {backbone_name}.')
        self.pos_enc = PositionalEncoding(self.d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(self.d_hidden, self.num_attention_heads, 2 * self.d_hidden)
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
        """Perform a forward pass.

        :param rgb_clip: Tensor of shape (Sequence Length, Batch Size, Channels, Height, Width).
        :return: Logits."""
        rgb_clip = batch.inputs
        t, b, c, h, w = rgb_clip.size()

        backbone_minibatch = rgb_clip.reshape(t * b, c, h, w)
        backbone_outputs = self.backbone(backbone_minibatch).reshape(t, b, -1)

        backbone_outputs = self.pos_enc(backbone_outputs)

        cls_inputs = torch.zeros((b, 1), dtype=torch.long).to(backbone_outputs.device)
        cls_embedded = self.cls_emb(cls_inputs).permute(1, 0, 2)
        self_attention_inputs = torch.cat((cls_embedded, backbone_outputs), dim=0)

        self_attention_outputs = self.self_attention(self_attention_inputs)

        classifier_inputs = self_attention_outputs[0]  # CLS token output.

        if self.language_embedding is not None:
            emb_lang = self.language_embedding(language_ids)
            classifier_inputs = torch.cat((classifier_inputs, emb_lang), dim=-1)

        logits = self.classifier(classifier_inputs)

        return logits

    def init_output_bias(self, class_weights: np.ndarray):
        self.classifier.bias.data = torch.log(class_weights)

    def setup_inference_hook(self, embedding_kind: str, hook_input, hook_output):
        if embedding_kind == 'spatial':
            self.pos_enc.register_forward_hook(hook_input)  # Input of positional encoding is the frame embedding.
        else:
            raise ValueError(f'Unsupported embedding kind {embedding_kind} for inference hook.')
