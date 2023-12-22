"""PyTorch Lightning module definition. Delegates computation to one of the defined networks."""
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .poseformer import PoseFormer
from .ptn import PTN
from .vtn import VTN


class Module(pl.LightningModule):
    """Pytorch Lightning module that delegates to neural networks.

    IT IS ASSUMED THAT ALL NETWORKS ARE BATCH_FIRST=FALSE!"""

    def __init__(self, model_name: str, batch_size: int, learning_rate: float, weight_decay: float,
                 num_attention_layers: int, num_attention_heads: int, d_hidden: int, num_classes: List[int],
                 weighted_loss: bool, use_bias_init: bool, disable_lr_scheduler: bool,
                 regularize_pose_embedding: bool, l1_lambda: float, regularize_attention: bool, l2_lambda: float,
                 backbone_name: str, **kwargs):
        """Create a Module.

        :param model_name: The name of the underlying neural network nn.Module.
        :param batch_size: Batch size.
        :param learning_rate: Initial learning rate.
        :param weight_decay: Weight decay.
        :param num_attention_layers: Number of self-attention layers.
        :param num_attention_heads: Number of self-attention heads in every layer.
        :param d_hidden: Size of the embeddings.
        :param num_classes: List of the number of classes in every language dataset.
        :param weighted_loss: If set to true, will compute class weights and use these in the CCE computation.
        :param use_bias_init: Initialize the bias of the network using the class distribution. May help convergence.
        :param disable_lr_scheduler: Disable the learning rate scheduler.
        :param regularize_pose_embedding: Enable L1 regularisation in the pose embedding's first layer.
        :param l1_lambda: The factor for L1 regularisation of the pose embedding.
        :param regularize_attention: Enable L2 regularisation of the self-attention network.
        :param l2_lambda: The factor for L2 regularisation of the self-attention network.
        :param backbone_name: (For image based networks, such as the VTN) the name of the visual backbone."""
        super(Module, self).__init__()

        # Hyperparameters / arguments.
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay if weight_decay is not None else 0.0
        self.num_classes = num_classes
        self.weighted_loss = weighted_loss
        self.use_bias_init = use_bias_init
        self.d_hidden = d_hidden
        self.disable_lr_scheduler = disable_lr_scheduler
        self.regularize_pose_embedding = regularize_pose_embedding
        self.l1_lambda = l1_lambda
        self.regularize_attention = regularize_attention
        self.l2_lambda = l2_lambda
        self.backbone_name = backbone_name
        self.retrain_on_all = kwargs['retrain_on_all']
        self.lr_steps = kwargs['lr_steps']
        self.pf_dropout = kwargs['pf_dropout']
        self.multilingual_conditioned = kwargs['multilingual_conditioned']
        self.lang_embedding_size = kwargs['lang_embedding_size']

        # Model initialization.
        if self.model_name == 'PTN':
            self.model = PTN(num_attention_layers, num_attention_heads, self.d_hidden, self.num_classes, **kwargs)
        elif self.model_name == 'PoseFormer':
            self.model = PoseFormer(kwargs['d_pose'], self.d_hidden, num_attention_layers, num_attention_heads,
                                    self.d_hidden * 2, dropout=self.pf_dropout, num_classes=self.num_classes,
                                    l1_label_norm=kwargs['l1_label_norm'], lang_embedding_size=self.lang_embedding_size,
                                    multilingual_conditioned=self.multilingual_conditioned)
        elif self.model_name == 'VTN':
            self.model = VTN(self.backbone_name, num_attention_layers, num_attention_heads, self.d_hidden,
                             self.num_classes)
        else:
            raise ValueError(f'Unknown model name {self.model_name}.')

        # Metrics.
        if not self.weighted_loss:
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
        else:
            self.class_weights = self._compute_class_weights(kwargs['data_dir']).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        if self.use_bias_init:
            self.prior_probabilities = self._compute_prior_probabilities(kwargs['data_dir']).to(self.device)
            self.model.init_output_bias(self.prior_probabilities)
        self.m_accuracy = torchmetrics.Accuracy()
        self.m_f1 = torchmetrics.F1Score(num_classes=int(np.sum(np.array(self.num_classes))), average="macro")

        # Save hyperparameters to model checkpoint.
        self.save_hyperparameters()

    def _compute_class_weights(self, data_dir: str) -> torch.Tensor:
        import os
        import csv
        import numpy as np
        from sklearn.utils import class_weight

        labels = []
        with open(os.path.join(data_dir, 'samples.csv'), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['Id', 'Label', 'Participant', 'Video', 'Subset']
            for row in reader:
                _id, label, _participant, video, subset = row
                if subset == 'train':
                    labels.append(int(label))

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        print('Computed class weights:')
        print(class_weights)
        return torch.from_numpy(class_weights).float()

    def _compute_prior_probabilities(self, data_dir: str) -> torch.Tensor:
        import os
        import csv
        import numpy as np

        labels = []
        with open(os.path.join(data_dir, 'samples.csv'), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['Id', 'Label', 'Participant', 'Video', 'Subset']
            for row in reader:
                _id, label, _participant, video, subset = row
                if subset == 'train':
                    labels.append(int(label))

        prior_probabilities = np.bincount(labels) / len(labels)
        print('Computed prior probabilities:')
        print(prior_probabilities)
        return torch.from_numpy(prior_probabilities).float()

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        # These if statements are used because different networks have different layer names.
        # They delete the classifier weights.
        if 'model.classifier.weight' in state_dict:
            del state_dict['model.classifier.weight']
            del state_dict['model.classifier.bias']
        else:
            del state_dict['model.head.weight']
            del state_dict['model.head.bias']
        if 'model.logit_norm.weight' in state_dict:
            del state_dict['model.logit_norm.weight']
            del state_dict['model.logit_norm.bias']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f'Loaded checkpoint {checkpoint_path}. The following keys were missing:')
        print(missing_keys)
        print('The following keys were unexpected:')
        print(unexpected_keys)

    def forward(self, inputs, language_ids):
        return self.model(inputs, language_ids)

    def training_step(self, batch, batch_idx):
        inputs, targets, _filenames, language_ids = batch
        z = self.model(inputs, language_ids)
        # Compute regularization weights.
        if self.regularize_pose_embedding and self.model_name == 'PTN':
            l1 = sum(p.abs().sum() for p in self.model.pose_embedding.embedding[0].parameters())
        elif self.regularize_pose_embedding and self.model_name == 'PoseFormer':
            l1 = sum(p.abs().sum() for p in self.model.frame_embed.proj[0].parameters())
        else:
            l1 = 0
        if self.regularize_attention:
            l2 = sum(torch.linalg.norm(p, 2) for layer in self.model.self_attention.layers for p in
                     layer.linear1.parameters()) + sum(
                torch.linalg.norm(p, 2) for layer in self.model.self_attention.layers for p in
                layer.linear2.parameters())
        else:
            l2 = 0
        # Compute total loss.
        loss = self.criterion(z, targets) + l1 * self.l1_lambda + l2 * self.l2_lambda

        preds = torch.argmax(z, dim=-1)
        batch_size = inputs.size(1)

        self.log('train_loss', loss, batch_size=batch_size)
        self.log('train_accuracy', self.m_accuracy(preds, targets), batch_size=batch_size)
        self.log('train_f1', self.m_f1(preds, targets), batch_size=batch_size)

        # Log accuracy per language.
        for i, nc in enumerate(self.num_classes):
            subset_preds = preds[torch.where(language_ids == i)]
            subset_targets = targets[torch.where(language_ids == i)]
            if len(subset_preds) > 0:
                self.log(f'train_accuracy_{i}', self.m_accuracy(subset_preds, subset_targets), batch_size=batch_size)
                self.log(f'train_f1_{i}', self.m_f1(subset_preds, subset_targets), batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _filenames, language_ids = batch
        z = self.model(inputs, language_ids)

        loss = self.criterion(z, targets)
        preds = torch.argmax(z, dim=-1)

        batch_size = inputs.size(1)
        self.log('val_loss', loss, batch_size=batch_size)
        self.log('val_accuracy', self.m_accuracy(preds, targets), batch_size=batch_size)
        self.log('val_f1', self.m_f1(preds, targets), batch_size=batch_size)

        # Compute accuracy per language.
        for i, nc in enumerate(self.num_classes):
            subset_preds = preds[torch.where(language_ids == i)]
            subset_targets = targets[torch.where(language_ids == i)]
            if len(subset_preds) > 0:
                self.log(f'val_accuracy_{i}', self.m_accuracy(subset_preds, subset_targets), batch_size=batch_size)
                self.log(f'val_f1_{i}', self.m_f1(subset_preds, subset_targets), batch_size=batch_size)

        # hp_metric for hyperparameter influence on accuracy tracking in tensorboard.
        self.log('hp_metric', self.m_accuracy(preds, targets), batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if not self.disable_lr_scheduler:
            if not self.retrain_on_all:
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5),
                    'monitor': 'val_accuracy'
                }
            else:
                return {
                    'optimizer': optimizer,
                    # When retrain-on-all is enabled, we use MultiStepLR to mimic ReduceLROnPlateau.
                    'lr_scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_steps, gamma=0.1),
                }
        else:
            return optimizer

    def log_grad_norm(self, grad_norm_dict):
        # Overwritten to support the `Batch` class.
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                      batch_size=self.batch_size)

    def freeze_part(self, parts: str) -> None:
        """Freeze parts of the model.

        :param parts: Comma separated list of fully qualified layer names."""
        if ',' in parts:
            for attr in parts.split(','):
                module = self.model.__getattr__(attr)
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                else:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        else:
            attr = parts
            module = self.model.__getattr__(attr)
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        # Module.
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--batch_size', type=int, help='Batch size.')
        parser.add_argument('--learning_rate', type=float, help='Base learning rate.')
        parser.add_argument('--weight_decay', type=float, help='Optimizer weight decay.')
        parser.add_argument('--num_classes', nargs='+', type=int, help='Number of classes, for every dataset.',
                            required=True)
        parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss.')
        parser.add_argument('--use_bias_init', action='store_true',
                            help='Initialize output layer bias to account for class imbalance.')
        parser.add_argument('--disable_lr_scheduler', action='store_true', help='Keep a constant learning rate.')
        # Transformers.
        parser.add_argument('--num_attention_layers', type=int, help='Number of multi-head attention layers.')
        parser.add_argument('--num_attention_heads', type=int, help='Number of attention heads per layer.')
        parser.add_argument('--regularize_attention', action='store_true',
                            help='Add L2 regularization to the feedforward layers in the transformer.')
        parser.add_argument('--l2_lambda', type=float, help='Strength of L2 regularization.', default=0.0)
        parser.add_argument('--l1_label_norm', action='store_true',
                            help='Apply logit normalisation and L1 regularisation of these logits.')
        # PoseFormer.
        parser.add_argument('--pf_dropout', type=float, default=0.2, help='PoseFormer dropout.')
        parser.add_argument('--multilingual_conditioned', action='store_true',
                            help='If using a multilingual model, enable language conditioning.')
        parser.add_argument('--lang_embedding_size', type=int, default=4,
                            help='(Conditioned multilingual models only) the dimensionality of the language embedding.')
        # Common.
        parser.add_argument('--d_hidden', type=int, help='Dimensionality of attention layers/LSTM hidden states.')
        # Image based models.
        parser.add_argument('--backbone_name', type=str, help='Feature extractor backbone name.')
        # Pose based models.
        parser.add_argument('--d_pose', type=int, help='Number of input features for pose data.')
        parser.add_argument('--regularize_pose_embedding', action='store_true',
                            help='Add L1 regularization to the initial layers of the pose embedder.')
        parser.add_argument('--l1_lambda', type=float, help='Strength of L1 regularization.', default=0.0)
        parser.add_argument('--residual_pose_embedding', action='store_true', help='Make the pose embedding residual.')
        parser.add_argument('--no_pose_embedding', action='store_true',
                            help='Disable pose embedding, simply add a linear layer after the poses.')
        parser.add_argument('--pose_embedding_kind', type=str, help='Kind of pose embedding (dense or attn)',
                            default='dense')

        return parent_parser
