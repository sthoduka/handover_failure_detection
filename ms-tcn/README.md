The original source repository for MS-TCN can be found [here](https://github.com/yabufarha/ms-tcn). The [original LICENSE](LICENSE_ORIGINAL) has been included here.

### Citation of the original code:
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    S. Li, Y. Abu Farha, Y. Liu, MM. Cheng,  and J. Gall.
    MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020

## Train and Test
Both variants (MSTCN-A and MSTCN-B) can be trained and evaluated using the following command:

```
python main.py \
  --train_root=/path/to/training_set \
  --train_labels=/path/to/training_labels \
  --validation_root=/path/to/val_set \
  --validation_labels=/path/to/val_labels \
  --test_root=/path/to/test_set \
  --test_labels=/path/to/test_labels \
  --loss_function=<loss function components> \
  --network_type=<network type> \
  --input_type=<input type> \
  --num_stages=2
  --n_threads=32 \
  --batch_size=4 \
  --default_root_dir=logs \
  --learning_rate=0.0005 \
  --max_epochs=40 \
  --log_every_n_steps=1 \
  --gpus=1
```

### Network Type
Set one of the following:
1. `--network_type='MSTCN-A'`
2. `--network_type='MSTCN-B'`

### Input Data Type
Set one of the following:
1. `--input_type='video'`
2. `--input_type='video_FT'`
3. `--input_type='video_gripper'`
4. `--input_type='video_FT_gripper'`

### Loss Function components
Set one of the following:
1. `--loss_function='Lcls'`: only outcome classification loss
2. `--loss_function='Lcls_LHSeg'`: outcome classification loss + human activity segmentation loss
3. `--loss_function='Lcls_LHSeg_LRseg'`: outcome classification loss + human activity segmentation loss + robot activity segmentation loss

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
