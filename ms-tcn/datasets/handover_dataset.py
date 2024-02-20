import os
import glob
import json
import numpy as np
import cv2
import pdb
import math

import torch

ALL = 'all'
r2h = 'robot to human handover'
h2r = 'human to robot handover'
KINOVA = 'Kinova Gen3'
HSR = 'Toyota HSR'


def get_gripper_state(robot_type, joint_states):
    '''
    -0.5: opened
    0.0: partially closed/open
    0.5: closed

    '''
    gripper_states = np.zeros(joint_states.shape[0])
    if robot_type == HSR:
        hand_motor = joint_states[7]
        gripper_states[joint_states[:, 7] > 0.95] = -0.5 # opened
        gripper_states[joint_states[:, 7] < -0.83] = 0.5 # closed
        # remaining = partially closed
    if robot_type == KINOVA:
        finger_joint = joint_states[7]
        gripper_states[joint_states[:, 7] < 0.1] = -0.5 # opened
        gripper_states[joint_states[:, 7] > 0.78] = 0.5 # closed
        # remaining = partially closed
    return gripper_states



def load_data(data_root, label_root=None, robot_type=ALL, task_type=ALL):
    trials = sorted(glob.glob(data_root + '/*'))
    data_robot_type = []
    data_task_type = []
    data_video_path = []
    data_wrench_aligned = []
    data_joint_pos_aligned = []
    data_phase = []
    data_label = []
    data_human_activity = []
    data_vid_feat = []
    data_flow_feat = []
    data_gripper_state = []
    data_trials = []
    for trial in trials:
        info_file = os.path.join(trial, 'task_info.json')
        with open(info_file, 'r') as fp:
            task_info = json.load(fp)
        if robot_type != ALL:
            if robot_type != task_info['robot']:
                continue
        if task_type != ALL:
            if task_type != task_info['task']:
                continue

        with open(os.path.join(label_root, os.path.basename(trial) + '.json')) as fp:
            label_data = json.load(fp)

        data_label.append(label_data['outcome'])
        data_trials.append(trial)
        data_robot_type.append(task_info['robot'])
        data_task_type.append(task_info['task'])

        data_video_path.append(os.path.join(trial, 'head_cam.mp4'))

        vid_feat = np.load(os.path.join(trial, 'i3d_kinetics_rgb.npy'))
        # training set contains features from multiple augmentations of the same video
        if os.path.exists(os.path.join(trial, 'i3d_kinetics_rgb_augmented.npy')):
            aug_vid_feat = np.load(os.path.join(trial, 'i3d_kinetics_rgb_augmented.npy'))
            vid_feat = np.concatenate((vid_feat[:, np.newaxis, :], aug_vid_feat), axis=1)
        else:
            vid_feat = vid_feat[:, np.newaxis, :]
        data_vid_feat.append(vid_feat)

        flow_feat = np.load(os.path.join(trial, 'i3d_kinetics_flow.npy'))
        # duplicate first frame
        flow_feat = np.concatenate((flow_feat[0, :][np.newaxis, :], flow_feat))
        data_flow_feat.append(flow_feat)
        phases = np.load(os.path.join(trial, 'robot_actions.npy'))

        aligned_wrench = np.load(os.path.join(trial, 'wrench_resampled.npy'))
        with open(os.path.join(os.path.dirname(data_root), 'wrench_stats.json'), 'r') as fp:
            wrench_stats = json.load(fp)
        mean = np.array(wrench_stats[task_info['robot']]['mean'])
        std = np.array(wrench_stats[task_info['robot']]['std'])
        aligned_wrench = np.divide((aligned_wrench - mean), std)
        data_wrench_aligned.append(aligned_wrench)

        aligned_joint_pos = np.load(os.path.join(trial, 'joint_pos_resampled.npy'))
        data_joint_pos_aligned.append(aligned_joint_pos)

        gripper_state = get_gripper_state(task_info['robot'], aligned_joint_pos)
        data_gripper_state.append(gripper_state)
        data_phase.append(phases)
        human_activity_state = np.load(os.path.join(trial, 'human_activity.npy'))

        data_human_activity.append(human_activity_state)
    data = {}
    data['robot'] = data_robot_type
    data['task'] = data_task_type
    data['video'] = data_video_path
    data['wrench_aligned'] = data_wrench_aligned
    data['label'] = data_label
    data['phase'] = data_phase
    data['joint_pos_aligned'] = data_joint_pos_aligned
    data['gripper_state'] = data_gripper_state
    data['human_activity'] = data_human_activity
    data['vid_feat'] = data_vid_feat
    data['flow_feat'] = data_flow_feat
    data['trials'] = data_trials
    return data

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label_root=None, robot_type=ALL, task_type=ALL, training=False):
        self.data_root = data_root
        self.data = load_data(data_root, label_root, robot_type, task_type)
        if len(self.data['video']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)
        self.maximum_length = 0
        human_activities = []
        for idx in range(len(self.data['video'])):
            robot_activity = self.data['phase'][idx]
            # we're not using idle and post-idle phases of the robot
            selected_indices = np.where((robot_activity > 0) & (robot_activity < 4))
            robot_activity = robot_activity[selected_indices]
            if len(robot_activity) > self.maximum_length:
                self.maximum_length = len(robot_activity)
        self.robot_type = robot_type
        self.task_type = task_type
        self.num_robot_classes = 7 # setting to same as number of human activities for convenience
        self.num_human_classes = 7
        self.training = training


    def __getitem__(self, index):
        robot_activity = torch.from_numpy(self.data['phase'][index])
        selected_indices = np.where((robot_activity > 0) & (robot_activity < 4))
        robot_activity = robot_activity[selected_indices]
        robot_activity = robot_activity - 1
        human_activity = torch.from_numpy(self.data['human_activity'][index][selected_indices])

        rand_idx = np.random.randint(self.data['vid_feat'][index].shape[1])
        feat = torch.from_numpy(self.data['vid_feat'][index][selected_indices][:, rand_idx, :])

        flow_feat = torch.from_numpy(self.data['flow_feat'][index][selected_indices])
        gripper_state = torch.from_numpy(self.data['gripper_state'][index][selected_indices]).unsqueeze(1).float()

        task = self.data['task'][index]
        if task == h2r:
            task_id = 0
        else:
            task_id = 1

        robot = self.data['robot'][index]
        if robot == HSR:
            robot_id = 0
        else:
            robot_id = 1

        task_variables = torch.as_tensor([self.data['label'][index], task_id, robot_id]).long()

        wrench = torch.from_numpy(self.data['wrench_aligned'][index][selected_indices]).float()

        feat = torch.cat((feat, flow_feat), axis=1)

        current_length = feat.shape[0]
        mask_robot = torch.ones(self.num_robot_classes, feat.shape[0], dtype=torch.float)
        mask_human = torch.ones(self.num_human_classes, feat.shape[0], dtype=torch.float)

        padding = self.maximum_length - current_length
        zeros = torch.zeros(padding, feat.shape[1], dtype=torch.float)
        feat = torch.cat((feat, zeros), axis=0)
        zeros = torch.zeros(self.num_robot_classes, padding, dtype=torch.float)
        mask_robot = torch.cat((mask_robot, zeros), axis=1)
        zeros = torch.zeros(self.num_human_classes, padding, dtype=torch.float)
        mask_human = torch.cat((mask_human, zeros), axis=1)

        zeros = torch.zeros(padding, wrench.shape[1], dtype=torch.float)
        wrench = torch.cat((wrench, zeros), axis=0)
        zeros = torch.zeros(padding, gripper_state.shape[1], dtype=torch.float)
        gripper_state = torch.cat((gripper_state, zeros), axis=0)

        ones = torch.ones(padding, dtype=torch.long) * (-100)
        robot_activity = torch.cat((robot_activity, ones))
        human_activity = torch.cat((human_activity, ones))

        feat = feat.permute(1, 0)
        wrench = wrench.permute(1, 0)
        gripper_state = gripper_state.permute(1, 0)

        path = self.data['video'][index]

        return feat, wrench, gripper_state, task_variables, robot_activity, human_activity, mask_robot, mask_human, path


    def __len__(self):
        return len(self.data['video'])

class StagedFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label_root=None, robot_type=ALL, task_type=ALL, clip_size=16, training=False):
        self.data_root = data_root
        self.data = load_data(data_root, label_root, robot_type, task_type)
        if len(self.data['video']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)

        self.maximum_length = 0
        human_activities = []
        for idx in range(len(self.data['video'])):
            robot_activity = self.data['phase'][idx]
            selected_indices = np.where((robot_activity > 0) & (robot_activity < 4))
            robot_activity = robot_activity[selected_indices]
            if len(robot_activity) > self.maximum_length:
                self.maximum_length = len(robot_activity)

        self.robot_type = robot_type
        self.task_type = task_type
        self.num_robot_classes = 7 # setting to same as number of human activities for convenience
        self.num_human_classes = 7
        self.training = training
        self.clip_size = clip_size


    def __getitem__(self, index):
        robot_activity = torch.from_numpy(self.data['phase'][index])
        phase1_idx = np.where(robot_activity == 1)[0] # approach
        phase2_idx = np.where(robot_activity == 2)[0] # interact
        phase3_idx = np.where(robot_activity == 3)[0] # retract

        replace = len(phase1_idx) < self.clip_size
        phase1_idx = np.random.choice(phase1_idx, self.clip_size, replace=replace)
        phase1_idx.sort()
        replace = len(phase2_idx) < self.clip_size
        phase2_idx = np.random.choice(phase2_idx, self.clip_size, replace=replace)
        phase2_idx.sort()
        replace = len(phase3_idx) < self.clip_size
        phase3_idx = np.random.choice(phase3_idx, self.clip_size, replace=replace)
        phase3_idx.sort()
        selected_indices = np.concatenate((phase1_idx, phase2_idx, phase3_idx))


        robot_activity = robot_activity[selected_indices]
        robot_activity -= 1

        human_activity = torch.from_numpy(self.data['human_activity'][index][selected_indices])

        rand_idx = np.random.randint(self.data['vid_feat'][index].shape[1])
        feat = torch.from_numpy(self.data['vid_feat'][index][selected_indices][:, rand_idx, :])

        flow_feat = torch.from_numpy(self.data['flow_feat'][index][selected_indices])
        gripper_state = torch.from_numpy(self.data['gripper_state'][index][selected_indices]).unsqueeze(1).float()

        task = self.data['task'][index]
        if task == h2r:
            task_id = 0
        else:
            task_id = 1

        robot = self.data['robot'][index]
        if robot == HSR:
            robot_id = 0
        else:
            robot_id = 1

        task_variables = torch.as_tensor([self.data['label'][index], task_id, robot_id]).long()


        wrench = torch.from_numpy(self.data['wrench_aligned'][index][selected_indices]).float()
        feat = torch.cat((feat, flow_feat), axis=1)

        mask_robot = torch.ones(self.num_robot_classes, feat.shape[0], dtype=torch.float)
        mask_human = torch.ones(self.num_human_classes, feat.shape[0], dtype=torch.float)

        feat = feat.permute(1, 0)
        wrench = wrench.permute(1, 0)
        gripper_state = gripper_state.permute(1, 0)

        path = self.data['video'][index]

        return feat, wrench, gripper_state, task_variables, robot_activity, human_activity, mask_robot, mask_human, path

    def __len__(self):
        return len(self.data['video'])
