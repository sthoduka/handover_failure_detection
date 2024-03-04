#!/usr/bin/python

# The original source for MultiStageModel, SingleStageModel and DilatedResidualLayer can be found here: https://github.com/yabufarha/ms-tcn/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pdb

class MultimodalClassifier(nn.Module):
    def __init__(self, hparams):
        super(MultimodalClassifier, self).__init__()
        self.mstcn = MultiStageModel(hparams)

        self.wrench_model = nn.Sequential(
                                nn.Conv1d(6, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                )
        self.gripper_model = nn.Sequential(
                                nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=2),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 16, kernel_size=3, padding=4, dilation=4),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                )
        self.hparams = hparams

    def forward(self, batch):
        feat, wrench, gripper, task_var, robot_activity, human_activity, mask_robot, mask_human, path = batch
        wrench_feat = self.wrench_model(wrench)
        gripper_feat = self.gripper_model(gripper)
        if self.hparams.input_type == 'video_FT_gripper':
            x = torch.cat((feat, wrench_feat, gripper_feat), axis=1)
        elif self.hparams.input_type == 'video_FT':
            x = torch.cat((feat, wrench_feat), axis=1)
        elif self.hparams.input_type == 'video_gripper':
            x = torch.cat((feat, gripper_feat), axis=1)
        elif self.hparams.input_type == 'video':
            x = feat
        mask = mask_human

        seg_result = self.mstcn(x, mask)
        if self.hparams.loss_function == 'Lcls':
            outputs = seg_result
            return outputs
        elif self.hparams.loss_function == 'Lcls_LHseg':
            outputs, outputs2 = seg_result
            return outputs, outputs2
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            outputs, outputs2, outputs3 = seg_result
            return outputs, outputs2, outputs3

    def loss_fn(self, out, batch):
        loss = self.mstcn.loss_fn(out, batch)
        return loss


class MultiStageModel(nn.Module):
    def __init__(self, hparams):
        super(MultiStageModel, self).__init__()
        if hparams.input_type == 'video_FT_gripper':
            dim = 2048 + 16 + 16
        elif hparams.input_type == 'video_FT':
            dim = 2048 + 16
        elif hparams.input_type == 'video_gripper':
            dim = 2048 + 16
        elif hparams.input_type == 'video':
            dim = 2048

        if hparams.loss_function == 'Lcls':
            self.stage1 = SingleStageModel(hparams.num_layers, hparams.num_f_maps, dim, hparams.num_outcome_classes-2, hparams)
            self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(hparams.num_layers, hparams.num_f_maps, hparams.num_outcome_classes-2, hparams.num_outcome_classes-2, hparams)) for s in range(hparams.num_stages-1)])
        elif hparams.loss_function == 'Lcls_LHseg':
            self.stage1 = SingleStageModel(hparams.num_layers, hparams.num_f_maps, dim, hparams.num_seg_classes, hparams)
            self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(hparams.num_layers, hparams.num_f_maps, hparams.num_outcome_classes-2+hparams.num_seg_classes, hparams.num_seg_classes, hparams)) for s in range(hparams.num_stages-1)])
        elif hparams.loss_function == 'Lcls_LHseg_LRseg':
            self.stage1 = SingleStageModel(hparams.num_layers, hparams.num_f_maps, dim, hparams.num_seg_classes, hparams)
            self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(hparams.num_layers, hparams.num_f_maps, hparams.num_outcome_classes-2+hparams.num_secondary_classes+hparams.num_seg_classes, hparams.num_seg_classes, hparams)) for s in range(hparams.num_stages-1)])

        # Input dropout layer is from ASFormer (https://arxiv.org/pdf/2110.08568.pdf)
        self.input_dropout = nn.Dropout2d()

        self.num_seg_classes = hparams.num_seg_classes
        self.hparams = hparams

        self.ce_seg_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, x, mask):
        x = x.unsqueeze(2)
        x = self.input_dropout(x)
        x = x.squeeze(2)

        if self.hparams.loss_function == 'Lcls':
            out = self.stage1(x, mask)
            outputs = out.unsqueeze(0)
            for s in self.stages:
                out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
                outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            return outputs
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            out1, out2, out3 = self.stage1(x, mask)
            outputs, outputs2, outputs3 = out1.unsqueeze(0), out2.unsqueeze(0), out3.unsqueeze(0)
            for s in self.stages:
                out = torch.cat((out1, out2, out3), axis=1)
                out1, out2, out3 = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
                outputs = torch.cat((outputs, out1.unsqueeze(0)), dim=0)
                outputs2 = torch.cat((outputs2, out2.unsqueeze(0)), dim=0)
                outputs3 = torch.cat((outputs3, out3.unsqueeze(0)), dim=0)
            return outputs, outputs2, outputs3
        elif self.hparams.loss_function == 'Lcls_LHseg':
            out1, out2 = self.stage1(x, mask)
            outputs, outputs2 = out1.unsqueeze(0), out2.unsqueeze(0)
            for s in self.stages:
                out = torch.cat((out1, out2), axis=1)
                out1, out2 = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
                outputs = torch.cat((outputs, out1.unsqueeze(0)), dim=0)
                outputs2 = torch.cat((outputs2, out2.unsqueeze(0)), dim=0)
            return outputs, outputs2


    def loss_fn(self, out, batch):
        feat, wrench, gripper, task_var, robot_activity, human_activity, mask_robot, mask_human, path = batch
        if self.hparams.loss_function == 'Lcls':
            outputs  = out
        elif self.hparams.loss_function == 'Lcls_LHseg':
            outputs, outputs2  = out
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            outputs, outputs2, outputs3  = out
        loss = 0
        for idx, p in enumerate(outputs):
            if self.hparams.loss_function != 'Lcls':
                loss += self.ce_seg_loss(p.transpose(2, 1).contiguous().view(-1, self.num_seg_classes), human_activity.view(-1))
                loss += 0.15*torch.mean(torch.clamp(self.mse_loss(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask_human[:, :, 1:])

            if self.hparams.loss_function == 'Lcls_LHseg_LRseg':
                loss += self.ce_seg_loss(outputs2[idx].transpose(2, 1).contiguous().view(-1, self.hparams.num_secondary_classes), robot_activity.view(-1))
                loss += 0.15*torch.mean(torch.clamp(self.mse_loss(F.log_softmax(outputs2[idx][:, :, 1:], dim=1), F.log_softmax(outputs2[idx].detach()[:, :, :-1], dim=1)), min=0, max=16)*mask_human[:, :self.hparams.num_secondary_classes, 1:])

                target = task_var[:, 0].unsqueeze(1).repeat(1, outputs3[idx].shape[2])
                target[robot_activity==-100] = -100
                target = target.view(-1)
                loss += self.ce_seg_loss(outputs3[idx].transpose(2, 1).contiguous().view(-1, self.hparams.num_outcome_classes-2), target)
            if self.hparams.loss_function == 'Lcls_LHseg':
                target = task_var[:, 0].unsqueeze(1).repeat(1, outputs2[idx].shape[2])
                target[robot_activity==-100] = -100
                target = target.view(-1)
                loss += self.ce_seg_loss(outputs2[idx].transpose(2, 1).contiguous().view(-1, self.hparams.num_outcome_classes-2), target)
            if self.hparams.loss_function == 'Lcls':
                target = task_var[:, 0].unsqueeze(1).repeat(1, outputs[idx].shape[2])
                target[robot_activity==-100] = -100
                target = target.view(-1)
                loss += self.ce_seg_loss(outputs[idx].transpose(2, 1).contiguous().view(-1, self.hparams.num_outcome_classes-2), target)


        return loss


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, hparams):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        if hparams.loss_function == 'Lcls_LHseg_LRseg':
            self.conv_out2 = nn.Conv1d(num_f_maps, hparams.num_secondary_classes, 1)
            self.conv_out3 = nn.Conv1d(num_f_maps, hparams.num_outcome_classes-2, 1)
        if hparams.loss_function == 'Lcls_LHseg':
            self.conv_out2 = nn.Conv1d(num_f_maps, hparams.num_outcome_classes-2, 1)
        self.hparams = hparams


    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        output = self.conv_out(out) * mask[:, 0:1, :]
        if self.hparams.loss_function == 'Lcls':
            return output
        elif self.hparams.loss_function == 'Lcls_LHseg':
            output2 = self.conv_out2(out) * mask[:, 0:1, :]
            return output, output2
        elif self.hparams.loss_function == 'Lcls_LHseg_LRseg':
            output2 = self.conv_out2(out) * mask[:, 0:1, :]
            output3 = self.conv_out3(out) * mask[:, 0:1, :]
            return output, output2, output3


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
