import argparse

import torch
from torchvision import transforms

import pdb

import numpy as np
import glob
import os
import cv2

from pytorch_i3d import InceptionI3d

def load_frames(imgs):
    frames = []
    frame_id = 0
    for img in imgs:
        xframe = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        yimg = img[:-5] + 'y.jpg'
        yframe = cv2.imread(yimg, cv2.IMREAD_GRAYSCALE)
        frame = np.stack((xframe, yframe), axis=2)
        frame = ((frame / 255.0) * 2) - 1.
        frames.append(frame)
        frame_id += 1
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='charades or kinetics')
    parser.add_argument('--root', type=str)
    args = parser.parse_args()

    i3d = InceptionI3d(400, in_channels=2)
    if args.model == 'charades':
        i3d.replace_logits(157)
        model_path = 'models/flow_charades.pt'
    elif args.model == 'kinetics':
        model_path = 'models/flow_imagenet.pt'
    else:
        print("invalid model")
        exit(0)

    i3d.load_state_dict(torch.load(model_path))
    i3d.cuda()
    i3d.train(False)

    image_transform = transforms.Compose([transforms.CenterCrop(480),
                                               transforms.Resize(224)])
    clip_length = 64
    trials = sorted(glob.glob(args.root + '/*'))
    for trial in trials:
        if os.path.exists(os.path.join(trial, 'i3d_%s_flow.npy' % args.model)):
            print('OF features already exist')
            exit(0)

        print(trial)
        imgs = sorted(glob.glob(os.path.join(trial, 'flow') + '/*_x.jpg'))

        if len(imgs) == 0:
            print('no flow images found')
            exit(0)

        video_frames = load_frames(imgs)
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
            clip = clip.permute(3, 0, 1, 2)
            clip = image_transform(clip)
            clip = clip.unsqueeze(0)
            clip = clip.cuda()
            with torch.no_grad():
                features = i3d.extract_features(clip)
            features = features.squeeze(3).squeeze(3).squeeze(0).mean(axis=1)
            all_features.append(features.detach().cpu().numpy())
        all_features = np.array(all_features)
        np.save(os.path.join(trial, 'i3d_%s_flow.npy' % args.model), all_features)


if __name__ == '__main__':
    main()
