import os
import glob
import json
import numpy as np
import cv2
import pdb

from torchvision import transforms
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
    data_gripper_state = []
    data_trials = []
    data_flow = []
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

        flow_files = np.array(sorted(glob.glob(os.path.join(trial, 'flow') + '/*_x.jpg')))
        data_flow.append(flow_files)

        phases = np.load(os.path.join(trial, 'robot_actions.npy'))
        aligned_wrench = np.load(os.path.join(trial, 'wrench_resampled.npy'))
        aligned_joint_pos = np.load(os.path.join(trial, 'joint_pos_resampled.npy'))

        with open(os.path.join(os.path.dirname(data_root), 'wrench_stats.json'), 'r') as fp:
            wrench_stats = json.load(fp)
        mean = np.array(wrench_stats[task_info['robot']]['mean'])
        std = np.array(wrench_stats[task_info['robot']]['std'])
        aligned_wrench = np.divide((aligned_wrench - mean), std)


        data_wrench_aligned.append(aligned_wrench)
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
    data['flow'] = data_flow
    data['wrench_aligned'] = data_wrench_aligned
    data['label'] = data_label
    data['phase'] = data_phase
    data['joint_pos_aligned'] = data_joint_pos_aligned
    data['gripper_state'] = data_gripper_state
    data['human_activity'] = data_human_activity
    data['trials'] = data_trials
    return data


def clip_loader_frame_ids(path, frame_ids):
    cap = cv2.VideoCapture(path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = int(num_frames)
    if num_frames == 0:
        print('this path has zero frames ', path)
    frame_id = 0
    frames = {}
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_id in frame_ids:
            frames[frame_id] = frame[:, :, ::-1].copy()
        frame_id += 1
    cap.release()
    missing = 0
    all_frames = []
    # makes sure we get duplicate frames too 
    for ff in frame_ids:
        if ff not in frames and len(all_frames) > 0:
            all_frames.append(all_frames[-1])
        elif ff not in frames:
            missing += 1
        else:
            all_frames.append(frames[ff])
    for idx in range(missing):
        all_frames.append(all_frames[-1])
    all_frames = np.array(all_frames)
    return all_frames

def flow_loader_frame_ids(flow_files, frame_ids):
    path_root = os.path.dirname(flow_files[0])
    all_frames = []
    for ff in frame_ids:
        if ff > len(flow_files):
            print("Trying to load optical flow file that doesn't exist. %d %s" % (ff, flow_files[-1]))
            exit(0)
        if ff == 0:
            flow_x = np.ones((480, 640), dtype=np.uint8) * 255
            flow_y = np.ones((480, 640), dtype=np.uint8) * 255
        else:
            flow_x = os.path.join(path_root, 'frame%04d_x.jpg' % ff)
            flow_y = os.path.join(path_root, 'frame%04d_y.jpg' % ff)
            flow_x = cv2.imread(flow_x, cv2.IMREAD_GRAYSCALE)
            flow_y = cv2.imread(flow_y, cv2.IMREAD_GRAYSCALE)
        flow = np.stack((flow_x, flow_y), axis=2)
        all_frames.append(flow)
    all_frames = np.array(all_frames)
    return all_frames


class FullVideoOutcomeDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label_root=None, robot_type=ALL, task_type=ALL, transform = None, evaluation=False, num_classes=4, video_type='rgb'):
        self.data_root = data_root
        self.data = load_data(data_root, label_root, robot_type, task_type)
        if len(self.data['video']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)

        self.robot_type = robot_type
        self.task_type = task_type
        self.evaluation = evaluation
        self.transform = transform
        self.clip_size = 64
        self.num_classes = num_classes
        self.video_type = video_type

    def __getitem__(self, index):
        video_id = index
        video_file = self.data['video'][video_id]
        phases = self.data['phase'][video_id]
        num_frames = len(phases)
        frame_ids = self.get_frame_ids(num_frames)

        if self.video_type == 'rgb':
            clip = clip_loader_frame_ids(video_file, frame_ids)
        elif self.video_type == 'flow':
            clip = flow_loader_frame_ids(self.data['flow'][video_id], frame_ids)
        if self.transform is not None:
            clip = torch.from_numpy(clip)
            clip = clip.permute((0, 3, 1, 2)).contiguous() # T, C, H, W
            clip = clip.to(dtype=torch.get_default_dtype()).div(255)
            clip = self.transform(clip)
            clip = clip.permute((1, 0, 2, 3)).contiguous() # C, T, H, W
        label = self.data['label'][video_id]

        return clip, label, video_id

    def get_frame_ids(self, num_frames):
        frame_ids = np.arange(0, num_frames, int(num_frames / self.clip_size))
        extra_frames = len(frame_ids) - self.clip_size
        start_id = extra_frames // 2
        frame_ids = frame_ids[start_id: start_id + self.clip_size]

        return frame_ids


    def __len__(self):
        return len(self.data['video'])


class MultimodalOutcomeDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label_root=None, robot_type=ALL, task_type=ALL, transform = None, evaluation=False, num_classes=4, video_type='rgb'):
        self.data_root = data_root
        self.data = load_data(data_root, label_root, robot_type, task_type)
        if len(self.data['video']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)

        self.robot_type = robot_type
        self.task_type = task_type
        self.evaluation = evaluation
        self.transform = transform
        self.clip_size = 64
        self.num_classes = num_classes
        self.video_type = video_type

    def __getitem__(self, index):
        video_id = index
        video_file = self.data['video'][video_id]

        wrench = self.data['wrench_aligned'][video_id]
        wrench = torch.from_numpy(wrench).float()

        gripper_state = self.data['gripper_state'][video_id]
        gripper_state = torch.from_numpy(gripper_state).unsqueeze(1).float()

        phases = self.data['phase'][video_id]
        phases_sampled = torch.from_numpy(phases).unsqueeze(1).float() - 2.5
        num_frames = len(phases)
        frame_ids = self.get_frame_ids(num_frames, self.clip_size)
        wrench = wrench[frame_ids]
        gripper_state = gripper_state[frame_ids]
        phases_sampled = phases_sampled[frame_ids]

        # T x C
        wrench = wrench.permute(1, 0)
        gripper_state = gripper_state.permute(1, 0)
        phases_sampled = phases_sampled.permute(1, 0)

        if self.video_type == 'rgb':
            clip = clip_loader_frame_ids(video_file, frame_ids)
        elif self.video_type == 'flow':
            clip = flow_loader_frame_ids(self.data['flow'][video_id], frame_ids)
        if self.transform is not None:
            clip = torch.from_numpy(clip)
            clip = clip.permute((0, 3, 1, 2)).contiguous() # T, C, H, W
            clip = clip.to(dtype=torch.get_default_dtype()).div(255)
            clip = self.transform(clip)
            clip = clip.permute((1, 0, 2, 3)).contiguous() # C, T, H, W
        label = self.data['label'][video_id]

        return clip, wrench, gripper_state, phases_sampled, label, video_id

    def get_frame_ids(self, num_frames, clip_length):
        frame_ids = np.arange(0, num_frames, int(num_frames / clip_length))
        extra_frames = len(frame_ids) - clip_length
        start_id = extra_frames // 2
        frame_ids = frame_ids[start_id: start_id + clip_length]

        return frame_ids

    def __len__(self):
        return len(self.data['video'])


class ProprioceptiveOutcomeDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, label_root=None, robot_type=ALL, task_type=ALL, transform = None, evaluation=False, num_classes=4):
        self.data_root = data_root
        self.data = load_data(data_root, label_root, robot_type, task_type)
        if len(self.data['video']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.data_root)
            raise RuntimeError(msg)

        self.robot_type = robot_type
        self.task_type = task_type
        self.evaluation = evaluation
        self.transform = transform
        self.clip_size = 64
        self.num_classes = num_classes

    def __getitem__(self, index):
        video_id = index

        wrench = self.data['wrench_aligned'][video_id]
        wrench = torch.from_numpy(wrench).float()

        gripper_state = self.data['gripper_state'][video_id]
        gripper_state = torch.from_numpy(gripper_state).unsqueeze(1).float()

        phases = self.data['phase'][video_id]
        phases_sampled = torch.from_numpy(phases).unsqueeze(1).float() - 2.5
        num_frames = len(phases)
        frame_ids = self.get_frame_ids(num_frames, self.clip_size)
        wrench = wrench[frame_ids]
        gripper_state = gripper_state[frame_ids]
        phases_sampled = phases_sampled[frame_ids]
        # T x C
        wrench = wrench.permute(1, 0)
        gripper_state = gripper_state.permute(1, 0)
        phases_sampled = phases_sampled.permute(1, 0)

        label = self.data['label'][video_id]

        return wrench, gripper_state, phases_sampled, label, video_id

    def get_frame_ids(self, num_frames, clip_length):
        frame_ids = np.arange(0, num_frames, int(num_frames / clip_length))
        extra_frames = len(frame_ids) - clip_length
        start_id = extra_frames // 2
        frame_ids = frame_ids[start_id: start_id + clip_length]

        return frame_ids



    def __len__(self):
        return len(self.data['video'])

def main():
    '''
    For testing only
    '''
    transform = transforms.Compose(
        [
            transforms.CenterCrop(448),
            transforms.Resize(224),
        ]
    )
    data_root = '/path/to/training_set'
    label_root = '/path/to/training_labels'
    dataset = FullVideoOutcomeDataset(data_root, label_root=label_root, robot_type=ALL, task_type=ALL, evaluation=False, transform=transform, video_type='rgb')
    clip, label, video_id = dataset[0]

    dataset = MultimodalOutcomeDataset(data_root, label_root=label_root, robot_type=ALL, task_type=ALL, evaluation=False, transform=transform, video_type='rgb')
    clip, wrench, gripper_state, phases, label, video_id = dataset[0]

    dataset = ProprioceptiveOutcomeDataset(data_root, label_root=label_root, robot_type=ALL, task_type=ALL, evaluation=False, transform=transform)
    wrench, gripper_state, phases, label, video_id = dataset[0]

if __name__ == '__main__':
    main()
