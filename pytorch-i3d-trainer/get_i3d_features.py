import argparse

import torch
import torchvision
from torchvision import transforms
import pdb

import numpy as np
import glob
import os
import cv2

from pytorch_i3d import InceptionI3d

def load_frames(video):
    frames = []
    cap = cv2.VideoCapture(video)
    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ((frame / 255.0) * 2) - 1.
            frames.append(frame)
            frame_id += 1
        else:
            break
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='charades or kinetics')
    parser.add_argument('--root', type=str)
    parser.add_argument('--augmented', action="store_true")
    args = parser.parse_args()
    i3d = InceptionI3d(400, in_channels=3)
    if args.model == 'charades':
        i3d.replace_logits(157)
        model_path = 'models/rgb_charades.pt'
    elif args.model == 'kinetics':
        model_path = 'models/rgb_imagenet.pt'
    else:
        print("invalid model")
        exit(0)

    i3d.load_state_dict(torch.load(model_path))
    i3d.cuda()
    i3d.train(False)

    if args.augmented:
        image_transform = transforms.Compose([transforms.RandomCrop(480),
                                              transforms.Resize(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(10)])
        filename = 'i3d_%s_rgb_augmented.npy' % args.model
    else:
        image_transform = transforms.Compose([transforms.CenterCrop(480),
                                                   transforms.Resize(224)])
        filename = 'i3d_%s_rgb.npy' % args.model

    clip_length = 64

    trials = sorted(glob.glob(args.root + '/*'))
    for trial in trials:
        if os.path.exists(os.path.join(trial, filename)):
            print('RGB features already exist')
            continue
        print(trial)
        video = os.path.join(trial, 'head_cam.mp4')
        video_frames = load_frames(video)
        all_features = []
        for frame_idx, frame in enumerate(video_frames):
            start_frame = max(0, frame_idx - clip_length)
            if frame_idx == 0:
                clip = [frame]
            else:
                clip = video_frames[start_frame:frame_idx]
            while len(clip) < clip_length:
                clip.insert(0, clip[0])

            clip = np.array(clip)
            clip = torch.from_numpy(clip).float()
            clip = clip.permute(0, 3, 1, 2) # T X C X H X W
            if args.augmented:
                trans_clips = None
                for idx in range(5):
                    trans_clip = image_transform(clip)
                    trans_clip = trans_clip.permute(1, 0, 2, 3) # C x T x H x W
                    trans_clip = trans_clip.unsqueeze(0)
                    trans_clip = trans_clip.cuda()
                    if trans_clips is None:
                        trans_clips = trans_clip
                    else:
                        trans_clips = torch.cat((trans_clips, trans_clip), axis=0)
                with torch.no_grad():
                    features = i3d.extract_features(trans_clips)
                    features = features.squeeze(3).squeeze(3).mean(axis=2)
                    all_features.append(features.detach().cpu().numpy())
            else:
                clip = image_transform(clip)
                clip = clip.permute(1, 0, 2, 3) # C X T x H x W
                clip = clip.unsqueeze(0)
                clip = clip.cuda()
                with torch.no_grad():
                    features = i3d.extract_features(clip)
                features = features.squeeze(3).squeeze(3).squeeze(0).mean(axis=1)
                all_features.append(features.detach().cpu().numpy())
        all_features = np.array(all_features)
        np.save(os.path.join(trial, filename), all_features)

if __name__ == '__main__':
    main()
