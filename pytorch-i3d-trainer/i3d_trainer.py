from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import lr_scheduler

from handover_dataset import FullVideoOutcomeDataset, MultimodalOutcomeDataset, ProprioceptiveOutcomeDataset
from pytorch_i3d import InceptionI3d, Unit3D
import torchnet.meter as meter

import pytorch_lightning as pl
import pdb
import numpy as np
import os

class VideoClassifier(nn.Module):
    def __init__(self, hparams, load_pretrained=False):
        super(VideoClassifier, self).__init__()
        if hparams.video_type == 'rgb':
            self.i3d = InceptionI3d(400, in_channels=3)
        else:
            self.i3d = InceptionI3d(400, in_channels=2)
        if load_pretrained:
            if hparams.model_path != '':
                print('Loading %s weights ' % hparams.model_path)
                self.i3d.load_state_dict(torch.load(hparams.model_path))
        self.i3d.replace_logits(hparams.num_classes)
        for params in self.i3d.parameters():
            params.requires_grad = False
        # unfreeze last logits layer
        for params in self.i3d.logits.parameters():
            params.requires_grad = True
        for params in self.i3d.avg_pool.parameters():
            params.requires_grad = True

    def forward(self, batch):
        inputs, labels, vidx = batch
        per_clip_logits = self.i3d(inputs)
        # batch x num_classes x 2 -> batch x num_classes
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        # batch x num_classes -> batch
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        inputs, labels, vidx = batch
        return labels

class MultimodalClassifier(nn.Module):
    def __init__(self, hparams, load_pretrained=False):
        super(MultimodalClassifier, self).__init__()
        if hparams.video_type == 'rgb':
            self.i3d = InceptionI3d(400, in_channels=3)
        else:
            self.i3d = InceptionI3d(400, in_channels=2)
        if load_pretrained:
            if hparams.model_path != '':
                print('Loading %s weights ' % hparams.model_path)
                self.i3d.load_state_dict(torch.load(hparams.model_path))
        self.i3d.replace_logits(hparams.num_classes, in_channels=1024+16+16)
        for params in self.i3d.parameters():
            params.requires_grad = False
        # unfreeze last logits layer
        for params in self.i3d.logits.parameters():
            params.requires_grad = True
        for params in self.i3d.avg_pool.parameters():
            params.requires_grad = True

        self.wrench_conv = nn.Sequential(
                                nn.Conv1d(6, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )
        self.gripper_conv = nn.Sequential(
                                nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )

    def forward(self, batch):
        inputs, wrench, gripper_state, phases, labels, vidx = batch
        i3d_feat = self.i3d.extract_features(inputs)
        wout = self.wrench_conv(wrench).unsqueeze(3).unsqueeze(3)
        gout = self.gripper_conv(gripper_state).unsqueeze(3).unsqueeze(3)
        feat = torch.cat((i3d_feat, wout, gout), axis=1)
        per_clip_logits = self.i3d.logits(self.i3d.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        inputs, wrench, gripper_state, phases, labels, vidx = batch
        return labels

class FTClassifier(nn.Module):
    def __init__(self, hparams):
        super(FTClassifier, self).__init__()
        self.wrench_conv = nn.Sequential(
                                nn.Conv1d(6, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )
        self.dropout = nn.Dropout(0.5)
        self.logits = Unit3D(in_channels=16, output_channels=hparams.num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')


    def forward(self, batch):
        wrench, gripper_state, phases, labels, vidx = batch
        wout = self.wrench_conv(wrench).unsqueeze(3).unsqueeze(3)
        feat = wout
        per_clip_logits = self.logits(self.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        wrench, gripper_state, phases, labels, vidx = batch
        return labels

class GripperClassifier(nn.Module):
    def __init__(self, hparams):
        super(GripperClassifier, self).__init__()
        self.gripper_conv = nn.Sequential(
                                nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )
        self.dropout = nn.Dropout(0.5)
        self.logits = Unit3D(in_channels=16, output_channels=hparams.num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')


    def forward(self, batch):
        wrench, gripper_state, phases, labels, vidx = batch
        gout = self.gripper_conv(gripper_state).unsqueeze(3).unsqueeze(3)
        feat = gout
        per_clip_logits = self.logits(self.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        wrench, gripper_state, phases, labels, vidx = batch
        return labels

class FTGripperClassifier(nn.Module):
    def __init__(self, hparams):
        super(FTGripperClassifier, self).__init__()
        self.wrench_conv = nn.Sequential(
                                nn.Conv1d(6, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )
        self.gripper_conv = nn.Sequential(
                                nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )
        self.dropout = nn.Dropout(0.5)
        self.logits = Unit3D(in_channels=32, output_channels=hparams.num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')


    def forward(self, batch):
        wrench, gripper_state, phases, labels, vidx = batch
        wout = self.wrench_conv(wrench).unsqueeze(3).unsqueeze(3)
        gout = self.gripper_conv(gripper_state).unsqueeze(3).unsqueeze(3)
        feat = torch.cat((wout, gout), axis=1)
        per_clip_logits = self.logits(self.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        wrench, gripper_state, phases, labels, vidx = batch
        return labels


class VideoFTClassifier(nn.Module):
    def __init__(self, hparams, load_pretrained=False):
        super(VideoFTClassifier, self).__init__()
        if hparams.video_type == 'rgb':
            self.i3d = InceptionI3d(400, in_channels=3)
        else:
            self.i3d = InceptionI3d(400, in_channels=2)
        if load_pretrained:
            if hparams.model_path != '':
                print('Loading %s weights ' % hparams.model_path)
                self.i3d.load_state_dict(torch.load(hparams.model_path))
        self.i3d.replace_logits(hparams.num_classes, in_channels=1024+16)
        for params in self.i3d.parameters():
            params.requires_grad = False

        # unfreeze last logits layer
        for params in self.i3d.logits.parameters():
            params.requires_grad = True
        for params in self.i3d.avg_pool.parameters():
            params.requires_grad = True

        self.wrench_conv = nn.Sequential(
                                nn.Conv1d(6, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )

    def forward(self, batch):
        inputs, wrench, gripper_state, phases, labels, vidx = batch
        i3d_feat = self.i3d.extract_features(inputs)
        wout = self.wrench_conv(wrench).unsqueeze(3).unsqueeze(3)
        feat = torch.cat((i3d_feat, wout), axis=1)
        per_clip_logits = self.i3d.logits(self.i3d.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        inputs, wrench, gripper_state, phases, labels, vidx = batch
        return labels

class VideoGripperClassifier(nn.Module):
    def __init__(self, hparams, load_pretrained=False):
        super(VideoGripperClassifier, self).__init__()
        if hparams.video_type == 'rgb':
            self.i3d = InceptionI3d(400, in_channels=3)
        else:
            self.i3d = InceptionI3d(400, in_channels=2)
        if load_pretrained:
            if hparams.model_path != '':
                print('Loading %s weights ' % hparams.model_path)
                self.i3d.load_state_dict(torch.load(hparams.model_path))
        self.i3d.replace_logits(hparams.num_classes, in_channels=1024+16)
        for params in self.i3d.parameters():
            params.requires_grad = False

        # unfreeze last logits layer
        for params in self.i3d.logits.parameters():
            params.requires_grad = True
        for params in self.i3d.avg_pool.parameters():
            params.requires_grad = True

        self.gripper_conv = nn.Sequential(
                                nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool1d(7)
                                )

    def forward(self, batch):
        inputs, wrench, gripper_state, phases, labels, vidx = batch
        i3d_feat = self.i3d.extract_features(inputs)
        gout = self.gripper_conv(gripper_state).unsqueeze(3).unsqueeze(3)
        feat = torch.cat((i3d_feat, gout), axis=1)
        per_clip_logits = self.i3d.logits(self.i3d.dropout(feat)).squeeze(3).squeeze(3)
        per_clip_logits = torch.max(per_clip_logits, dim=2)[0]
        predictions = torch.argmax(per_clip_logits, axis=1)
        return per_clip_logits, predictions

    def get_labels(self, batch):
        inputs, wrench, gripper_state, phases, labels, vidx = batch
        return labels


class i3DTrainer(pl.LightningModule):
    def __init__(self, hparams, load_pretrained=False):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.classifier_type == 'video':
            self.model = VideoClassifier(self.hparams, load_pretrained)
        elif self.hparams.classifier_type == 'multimodal':
            self.model = MultimodalClassifier(self.hparams, load_pretrained)
        elif self.hparams.classifier_type == 'FT_only':
            self.model = FTClassifier(self.hparams)
        elif self.hparams.classifier_type == 'gripper_only':
            self.model = GripperClassifier(self.hparams)
        elif self.hparams.classifier_type == 'FTgripper_only':
            self.model = FTGripperClassifier(self.hparams)
        elif self.hparams.classifier_type == 'video_FT':
            self.model = VideoFTClassifier(self.hparams, load_pretrained)
        elif self.hparams.classifier_type == 'video_gripper':
            self.model = VideoGripperClassifier(self.hparams, load_pretrained)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.map_meter = meter.mAPMeter()
        self.cls_meter = meter.APMeter()
        self.conf_meter = meter.ConfusionMeter(self.hparams.num_classes)

    def forward(self, batch):
        return self.model(batch)

    def calculate_loss(self, batch, per_frame_logits):
        labels = self.model.get_labels(batch)
        loss = self.loss_fn(per_frame_logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        per_frame_logits, predictions = self(batch)
        loss = self.calculate_loss(batch, per_frame_logits)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        per_clip_logits, predictions = self(batch)
        loss = self.calculate_loss(batch, per_clip_logits)

        labels = self.model.get_labels(batch)
        label_one_hot = F.one_hot(labels, num_classes=self.hparams.num_classes)
        self.map_meter.add(per_clip_logits, label_one_hot)
        self.cls_meter.add(per_clip_logits, label_one_hot)
        self.conf_meter.add(per_clip_logits, label_one_hot)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        per_clip_logits, predictions = self(batch)
        loss = self.calculate_loss(batch, per_clip_logits)

        labels = self.model.get_labels(batch)
        label_one_hot = F.one_hot(labels, num_classes=self.hparams.num_classes)
        self.map_meter.add(per_clip_logits, label_one_hot)
        self.cls_meter.add(per_clip_logits, label_one_hot)
        self.conf_meter.add(per_clip_logits, label_one_hot)
        return {'loss': loss, 'gt': labels, 'logits': per_clip_logits}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

        mAP = self.map_meter.value()
        AP = self.cls_meter.value()
        for idx, cls_ap in enumerate(AP):
            self.log("val_AP_%02d" % idx, cls_ap)

        self.log("val_mAP", mAP)
        conf = self.conf_meter.value()
        acc = np.diag(conf).sum() / conf.sum()
        self.log("val_acc", acc)
        self.map_meter.reset()
        self.cls_meter.reset()
        self.conf_meter.reset()

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("test_loss", avg_loss)

        gt = torch.cat([x['gt'] for x in outputs]).detach().cpu().numpy()
        logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy()
        np.save(os.path.join(self.logger.log_dir, 'gt.npy'), gt)
        np.save(os.path.join(self.logger.log_dir, 'logits.npy'), logits)

        mAP = self.map_meter.value()
        AP = self.cls_meter.value()
        for idx, cls_ap in enumerate(AP):
            self.log("test_AP_%02d" % idx, cls_ap)

        self.log("test_mAP", mAP)
        conf = self.conf_meter.value()
        acc = np.diag(conf).sum() / conf.sum()
        self.log("test_acc", acc)
        self.map_meter.reset()
        self.cls_meter.reset()
        self.conf_meter.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        clip_transform = transforms.Compose([transforms.RandomCrop(448),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize(224)])
        if self.hparams.classifier_type == 'video':
            train_dataset = FullVideoOutcomeDataset(self.hparams.train_root, self.hparams.train_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         video_type=self.hparams.video_type,
                                         robot_type=self.hparams.robot_type_trainval, task_type=self.hparams.task_type_trainval)
        elif self.hparams.classifier_type == 'multimodal' or self.hparams.classifier_type == 'video_gripper' or self.hparams.classifier_type == 'video_FT':
            train_dataset = MultimodalOutcomeDataset(self.hparams.train_root, self.hparams.train_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         video_type=self.hparams.video_type,
                                         robot_type=self.hparams.robot_type_trainval, task_type=self.hparams.task_type_trainval)
        elif self.hparams.classifier_type == "FT_only" or self.hparams.classifier_type == 'gripper_only' or self.hparams.classifier_type == 'FTgripper_only':
            train_dataset = ProprioceptiveOutcomeDataset(self.hparams.train_root, self.hparams.train_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         robot_type=self.hparams.robot_type_trainval, task_type=self.hparams.task_type_trainval)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.n_threads, pin_memory=True)
    def val_dataloader(self):
        clip_transform = transforms.Compose([transforms.CenterCrop(448),
                                             transforms.Resize(224)])
        if self.hparams.classifier_type == 'video':
            dataset = FullVideoOutcomeDataset(self.hparams.validation_root, self.hparams.validation_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         video_type=self.hparams.video_type,
                                         robot_type=self.hparams.robot_type_trainval, task_type=self.hparams.task_type_trainval)
        elif self.hparams.classifier_type == 'multimodal' or self.hparams.classifier_type == 'video_gripper' or self.hparams.classifier_type == 'video_FT':
            dataset = MultimodalOutcomeDataset(self.hparams.validation_root, self.hparams.validation_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         video_type=self.hparams.video_type,
                                         robot_type=self.hparams.robot_type_trainval, task_type=self.hparams.task_type_trainval)
        elif self.hparams.classifier_type == "FT_only" or self.hparams.classifier_type == 'gripper_only' or self.hparams.classifier_type == 'FTgripper_only':
            dataset = ProprioceptiveOutcomeDataset(self.hparams.validation_root, self.hparams.validation_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         robot_type=self.hparams.robot_type_trainval, task_type=self.hparams.task_type_trainval)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=True)

    def test_dataloader(self):
        clip_transform = transforms.Compose([transforms.CenterCrop(448),
                                             transforms.Resize(224)])
        if self.hparams.classifier_type == 'video':
            dataset = FullVideoOutcomeDataset(self.hparams.test_root, self.hparams.test_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         video_type=self.hparams.video_type,
                                         robot_type=self.hparams.robot_type_test, task_type=self.hparams.task_type_test)
        elif self.hparams.classifier_type == 'multimodal' or self.hparams.classifier_type == 'video_gripper' or self.hparams.classifier_type == 'video_FT':
            dataset = MultimodalOutcomeDataset(self.hparams.test_root, self.hparams.test_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         video_type=self.hparams.video_type,
                                         robot_type=self.hparams.robot_type_test, task_type=self.hparams.task_type_test)
        elif self.hparams.classifier_type == "FT_only" or self.hparams.classifier_type == 'gripper_only' or self.hparams.classifier_type == 'FTgripper_only':
            dataset = ProprioceptiveOutcomeDataset(self.hparams.test_root, self.hparams.test_labels,
                                         transform=clip_transform, num_classes=self.hparams.num_classes,
                                         robot_type=self.hparams.robot_type_test, task_type=self.hparams.task_type_test)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=True)
