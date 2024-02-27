The original source repository for the `pytorch_i3d` model and the pre-trained weights under [models](models/) can be found [here](https://github.com/piergiaj/pytorch-i3d). The [original LICENSE](LICENSE_ORIGINAL.txt) has been included here.

## Train and Test
All four variants (for both RGB and optical flow inputs) can be trained and evaluated using the following command:

```
python main.py \
  --train_root=/path/to/training_set \
  --train_labels=/path/to/training_labels \
  --validation_root=/path/to/val_set \
  --validation_labels=/path/to/val_labels \
  --test_root=/path/to/test_set \
  --test_labels=/path/to/test_labels \
  --model_path=models/<video_type>_imagenet.pt \
  --classifier_type=<classifier_type> \
  --video_type=<video_type> \
  --num_classes=4 \
  --batch_size=8 \
  --default_root_dir=logs/ \
  --learning_rate=0.001 \
  --max_epochs=40 \
  --log_every_n_steps=1 \
  --gpus=1
```
### Modalities
Select one of the following:
1. `classifier_type=video`
2. `classifier_type=multimodal` (all three modalities)
3. `classifier_type=FT_only`
4. `classifier_type=gripper_only`
5. `classifier_type=FTgripper_only`
6. `classifier_type=video_FT`
7. `classifier_type=video_gripper`


For late fusion in each of the following variants, sum the logits from the individual networks at test time to get the final result.

### I3D-A
Separately train the following networks
1. I3D RGB (`classifier_type=video`, `model_path=models/rgb_imagenet.pt`, `video_type=rgb`)
2. I3D Flow (`classifier_type=video`, `model_path=models/flow_imagenet.pt`, `video_type=flow`)
3. FT (`classifier_type=FT_only`)
4. Gripper (`classifier_type=gripper_only`)

### I3D-B
Separately train the following networks:
1. I3D RGB+FT+Gripper (`classifier_type=multimodal`, `model_path=models/rgb_imagenet.pt`, `video_type=rgb`)
2. I3D Flow (`classifier_type=video`, `model_path=models/flow_imagenet.pt`, `video_type=flow`)

### I3D-C
Separately train the following networks:
1. I3D RGB (`classifier_type=video`, `model_path=models/rgb_imagenet.pt`, `video_type=rgb`)
2. I3D Flow+FT+Gripper (`classifier_type=multimodal`, `model_path=models/flow_imagenet.pt`, `video_type=flow`)

### I3D-D
Separately train the following networks for the full versions:
1. I3D RGB+FT+Gripper (`classifier_type=multimodal`, `model_path=models/rgb_imagenet.pt`, `video_type=rgb`)
2. I3D Flow+FT+Gripper (`classifier_type=multimodal`, `model_path=models/flow_imagenet.pt`, `video_type=flow`)

The paper also reports results using the `classifier_type=video_FT` and `classifier_type=video_gripper` (rows 9 and 10 in Table III).

### Robot Type
To limit training to one of the robot types, use the `robot_type_trainval` argument as follows:
1. HSR only: `robot_type_trainval="Toyota HSR"`
2. Kinova only: `robot_type_trainval="Kinova Gen3"`

To evaluate only on one of the robot types, use:
1. HSR only: `robot_type_test="Toyota HSR"`
2. Kinova only: `robot_type_test="Kinova Gen3"`

### Task Type
To train and evaluate each handover task separately, use:
1. Human to Robot Handover only: `task_type_trainval="human to robot handover"` and `task_type_test="human to robot handover"`
2. Robot to Human Handover only: `task_type_trainval="robot to human handover"` and `task_type_test="robot to human handover"`

## Extract features
The dataset already contains the I3D features for both RGB and optical flow. In case you want to extract them again, use the following commands. All use the I3D model pre-trained on the Kinetics dataset.


Extract features for RGB frames:

```
python get_i3d_features.py --root /path/to/training_set --model kinetics
```

Extract features for augmented RGB frames:

```
python get_i3d_features.py --root /path/to/training_set --model kinetics --augmented
```

Extract features for optical flow frames:

```
python get_i3d_features_optical_flow.py --root /path/to/training_set --model kinetics
```
