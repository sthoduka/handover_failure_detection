from models.mstcn import MultimodalClassifier
from datasets.handover_dataset import FeatureDataset, StagedFeatureDataset
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import pdb
import os
from metrics import f_score, edit_score
import torchnet.meter as meter

class SegmentationTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = MultimodalClassifier(self.hparams)
        self.conf_meter = meter.ConfusionMeter(self.hparams.num_outcome_classes - 2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # MS-TCN params
        parser.add_argument('--num_stages', type=int, default=3, help='number of stages')
        parser.add_argument('--num_layers', type=int, default=10, help='number of layers per stage')
        parser.add_argument('--num_f_maps', type=int, default=64, help='number of feature maps')

        # fixed params
        parser.add_argument('--num_seg_classes', type=int, default=7, help='number of classes for segmentation of human activity')
        parser.add_argument('--num_secondary_classes', type=int, default=3, help='number of classes for segmentation for robot activity (if loss function includes LRseg)')
        parser.add_argument('--num_outcome_classes', type=int, default=6, help='number of classes for classification')

        # variable params
        parser.add_argument('--loss_function', type=str, default='Lcls_LHseg', help='Lcls, Lcls_LHseg, Lcls_LHseg_LRseg')
        parser.add_argument('--network_type', type=str, default='MSTCN-B', help='MSTCN-B or MSTCN-A')
        parser.add_argument('--input_type', type=str, default='video', help='video, video_FT, video_gripper or video_FT_gripper')
        parser.add_argument('--robot_type_trainval', type=str, default='all', help='all, Toyota HSR or Kinova Gen3 for training and validation')
        parser.add_argument('--task_type_trainval', type=str, default='all', help='all, human to robot handover, robot to human handover for training and validation')
        parser.add_argument('--robot_type_test', type=str, default='all', help='all, Toyota HSR or Kinova Gen3 for testing')
        parser.add_argument('--task_type_test', type=str, default='all', help='all, human to robot handover, robot to human handover for testing')
        return parser

    def forward(self, batch):
        feat, wrench, gripper, task_var, robot_activity, human_activity, mask_robot, mask_human, path = batch
        out = self.model(batch)
        return out

    def loss_function(self, out, batch):
        return self.model.loss_fn(out, batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_function(out, batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_function(out, batch)
        feat, wrench, gripper, task_var, robot_activity, human_activity, mask_robot, mask_human, path = batch
        if self.hparams.loss_function == 'Lcls':
            outputs2 = out
            _, predictions2 = torch.max(outputs2[-1], 1)
            per_vid_predictions2 = []
            per_vid_gt2 = []
            last_pred = []
            for vididx, pred in enumerate(predictions2):
                per_vid_predictions2.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt2.append(task_var[vididx, 0])
                last_pred.append(per_vid_predictions2[-1][-1])
            last_pred = torch.stack(last_pred)
            self.conf_meter.add(last_pred, task_var[:, 0])
        elif self.hparams.loss_function == 'Lcls_LHseg':
            outputs, outputs2 = out
            _, predictions2 = torch.max(outputs2[-1], 1)
            per_vid_predictions2 = []
            per_vid_gt2 = []
            last_pred = []
            for vididx, pred in enumerate(predictions2):
                per_vid_predictions2.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt2.append(task_var[vididx, 0])
                last_pred.append(per_vid_predictions2[-1][-1])
            last_pred = torch.stack(last_pred)
            self.conf_meter.add(last_pred, task_var[:, 0])
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            outputs, outputs2, outputs3 = out
            _, predictions2 = torch.max(outputs2[-1], 1)
            _, predictions3 = torch.max(outputs3[-1], 1)
            per_vid_predictions2 = []
            per_vid_gt2 = []
            for vididx, pred in enumerate(predictions2):
                per_vid_predictions2.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt2.append(robot_activity[vididx][mask_human[vididx, 0, :] == 1].cpu())
            per_vid_predictions3 = []
            per_vid_gt3 = []
            last_pred = []
            for vididx, pred in enumerate(predictions3):
                per_vid_predictions3.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt3.append(task_var[vididx, 0])
                last_pred.append(per_vid_predictions3[-1][-1])
            last_pred = torch.stack(last_pred)
            self.conf_meter.add(last_pred, task_var[:, 0])


        if self.hparams.loss_function != 'Lcls':
            _, predictions = torch.max(outputs[-1], 1)
            per_vid_predictions = []
            per_vid_gt = []
            for vididx, pred in enumerate(predictions):
                per_vid_predictions.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt.append(human_activity[vididx][mask_human[vididx, 0, :] == 1].cpu())

        if self.hparams.loss_function == 'Lcls':
            return {'loss': loss}
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            return {'loss': loss, 'predictions': per_vid_predictions, 'gt': per_vid_gt, 'robot_predictions': per_vid_predictions2, 'robot_gt': per_vid_gt2, 'cls_predictions': per_vid_predictions3, 'cls_gt': per_vid_gt3}
        elif self.hparams.loss_function == 'Lcls_LHseg':
            return {'loss': loss, 'predictions': per_vid_predictions, 'gt': per_vid_gt, 'cls_predictions': per_vid_predictions2, 'cls_gt': per_vid_gt2}

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_function(out, batch)
        feat, wrench, gripper, task_var, robot_activity, human_activity, mask_robot, mask_human, path = batch
        if self.hparams.loss_function == 'Lcls':
            outputs2 = out
            _, predictions2 = torch.max(outputs2[-1], 1)
            per_vid_predictions2 = []
            per_vid_gt2 = []
            last_pred = []
            for vididx, pred in enumerate(predictions2):
                per_vid_predictions2.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt2.append(task_var[vididx, 0])
                last_pred.append(per_vid_predictions2[-1][-1])
            last_pred = torch.stack(last_pred)
            self.conf_meter.add(last_pred, task_var[:, 0])
        elif self.hparams.loss_function == 'Lcls_LHseg':
            outputs, outputs2 = out
            _, predictions2 = torch.max(outputs2[-1], 1)
            per_vid_predictions2 = []
            per_vid_gt2 = []
            last_pred = []
            for vididx, pred in enumerate(predictions2):
                per_vid_predictions2.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt2.append(task_var[vididx, 0])
                last_pred.append(per_vid_predictions2[-1][-1])
            last_pred = torch.stack(last_pred)
            self.conf_meter.add(last_pred, task_var[:, 0])
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            outputs, outputs2, outputs3 = out
            _, predictions2 = torch.max(outputs2[-1], 1)
            _, predictions3 = torch.max(outputs3[-1], 1)
            per_vid_predictions2 = []
            per_vid_gt2 = []
            for vididx, pred in enumerate(predictions2):
                per_vid_predictions2.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt2.append(robot_activity[vididx][mask_human[vididx, 0, :] == 1].cpu())
            per_vid_predictions3 = []
            per_vid_gt3 = []
            last_pred = []
            for vididx, pred in enumerate(predictions3):
                per_vid_predictions3.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt3.append(task_var[vididx, 0])
                last_pred.append(per_vid_predictions3[-1][-1])
            last_pred = torch.stack(last_pred)
            self.conf_meter.add(last_pred, task_var[:, 0])

        if self.hparams.loss_function != 'Lcls':
            _, predictions = torch.max(outputs[-1], 1)
            per_vid_predictions = []
            per_vid_gt = []
            for vididx, pred in enumerate(predictions):
                per_vid_predictions.append(pred[mask_human[vididx, 0, :] == 1].cpu())
                per_vid_gt.append(human_activity[vididx][mask_human[vididx, 0, :] == 1].cpu())

        if self.hparams.loss_function == 'Lcls':
            return {'loss': loss}
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            return {'loss': loss, 'predictions': per_vid_predictions, 'gt': per_vid_gt, 'robot_predictions': per_vid_predictions2, 'robot_gt': per_vid_gt2, 'cls_predictions': per_vid_predictions3, 'cls_gt': per_vid_gt3}
        elif self.hparams.loss_function == 'Lcls_LHseg':
            return {'loss': loss, 'predictions': per_vid_predictions, 'gt': per_vid_gt, 'cls_predictions': per_vid_predictions2, 'cls_gt': per_vid_gt2}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.hparams.loss_function != 'Lcls':
            accuracy, edit, f1_scores = self.compute_segmentation_metrics(outputs, 'predictions', 'gt')
        if self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            robot_accuracy, robot_edit, robot_f1_scores = self.compute_segmentation_metrics(outputs, 'robot_predictions', 'robot_gt')
            self.log("val_seg_acc_robot", robot_accuracy)
            self.log("val_edit_score_robot", robot_edit)
            self.log("val_f1@10_robot", robot_f1_scores[0])
            self.log("val_f1@25_robot", robot_f1_scores[1])
            self.log("val_f1@50_robot", robot_f1_scores[2])


        conf_matrix2 = self.conf_meter.value()
        cls_accuracy2 = np.trace(conf_matrix2) / np.sum(conf_matrix2)
        self.log("val_cls_acc2", cls_accuracy2)
        self.conf_meter.reset()

        self.log("val_loss", avg_loss)
        if self.hparams.loss_function != 'Lcls':
            self.log("val_seg_acc", accuracy)
            self.log("val_edit_score", edit)
            self.log("val_f1@10", f1_scores[0])
            self.log("val_f1@25", f1_scores[1])
            self.log("val_f1@50", f1_scores[2])

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if 'cls_predictions' in outputs[0]:
            gt = torch.cat([torch.stack(x['cls_gt']) for x in outputs]).detach().cpu().numpy()
            preds = []
            for x in outputs:
                pred_batch = x['cls_predictions']
                for pp in pred_batch:
                    preds.append(pp[-1])
            preds = torch.stack(preds).numpy()
            np.save(os.path.join(self.logger.log_dir, 'gt.npy'), gt)
            np.save(os.path.join(self.logger.log_dir, 'predictions.npy'), preds)

        if self.hparams.loss_function != 'Lcls':
            np_predictions = []
            np_gt = []
            for batch in outputs:
                for bb in batch['predictions']:
                    np_predictions.append(bb.numpy())
                for bb in batch['gt']:
                    np_gt.append(bb.numpy())
            gt_segmentation = np.array(np_gt, dtype=object)
            pred_segmentation = np.array(np_predictions, dtype=object)
            np.save(os.path.join(self.logger.log_dir, 'gt_seg.npy'), gt_segmentation)
            np.save(os.path.join(self.logger.log_dir, 'pred_seg.npy'), pred_segmentation)

            accuracy, edit, f1_scores = self.compute_segmentation_metrics(outputs, 'predictions', 'gt')
        if self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            robot_accuracy, robot_edit, robot_f1_scores = self.compute_segmentation_metrics(outputs, 'robot_predictions', 'robot_gt')
            self.log("test_seg_acc_robot", robot_accuracy)
            self.log("test_edit_score_robot", robot_edit)
            self.log("test_f1@10_robot", robot_f1_scores[0])
            self.log("test_f1@25_robot", robot_f1_scores[1])
            self.log("test_f1@50_robot", robot_f1_scores[2])

        conf_matrix2 = self.conf_meter.value()
        cls_accuracy2 = np.trace(conf_matrix2) / np.sum(conf_matrix2)
        self.log("test_cls_acc2", cls_accuracy2)
        self.conf_meter.reset()

        self.log("test_loss", avg_loss)
        if self.hparams.loss_function != 'Lcls':
            self.log("test_seg_acc", accuracy)
            self.log("test_edit_score", edit)
            self.log("test_f1@10", f1_scores[0])
            self.log("test_f1@25", f1_scores[1])
            self.log("test_f1@50", f1_scores[2])

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def compute_segmentation_metrics(self, outputs, pred_key, gt_key):
        '''
        See https://github.com/yabufarha/ms-tcn/blob/master/eval.py for original source
        '''
        correct = 0
        total = 0
        edit = 0
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        overlap = [.1, .25, .5]
        for batch in outputs:
            for vidx in range(len(batch[pred_key])):
                pred = batch[pred_key][vidx]
                gt = batch[gt_key][vidx]
                correct += (pred == gt).sum().item()
                total += len(gt)

                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(pred, gt, overlap[s], bg_class=[])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                edit += edit_score(pred, gt, bg_class=[])

        accuracy = correct / total
        edit = edit / total
        f1_scores = []
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            f1_scores.append(f1)
        return accuracy, edit, f1_scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        if self.hparams.network_type == 'MSTCN-B':
            train_dataset = FeatureDataset(self.hparams.train_root, self.hparams.train_labels, self.hparams.robot_type_trainval, self.hparams.task_type_trainval)
        elif self.hparams.network_type == 'MSTCN-A':
            train_dataset = StagedFeatureDataset(self.hparams.train_root, self.hparams.train_labels, self.hparams.robot_type_trainval, self.hparams.task_type_trainval)
        if self.training:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=True, num_workers=self.hparams.n_threads, pin_memory=False)
        else:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

    def val_dataloader(self):
        if self.hparams.network_type == 'MSTCN-B':
            val_dataset = FeatureDataset(self.hparams.validation_root, self.hparams.validation_labels, self.hparams.robot_type_trainval, self.hparams.task_type_trainval)
        elif self.hparams.network_type == 'MSTCN-A':
            val_dataset = StagedFeatureDataset(self.hparams.validation_root, self.hparams.validation_labels, self.hparams.robot_type_trainval, self.hparams.task_type_trainval)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)
    def test_dataloader(self):
        if self.hparams.network_type == 'MSTCN-B':
            test_dataset = FeatureDataset(self.hparams.test_root, self.hparams.test_labels, self.hparams.robot_type_test, self.hparams.task_type_test)
        elif self.hparams.network_type == 'MSTCN-A':
            test_dataset = StagedFeatureDataset(self.hparams.test_root, self.hparams.test_labels, self.hparams.robot_type_test, self.hparams.task_type_test)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=False)

