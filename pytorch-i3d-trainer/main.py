import pytorch_lightning as pl
from argparse import ArgumentParser
from i3d_trainer import i3DTrainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    parser = ArgumentParser()

    parser.add_argument('--train_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--train_labels', default='', type=str, help='Path to training labels')
    parser.add_argument('--validation_root', default='', type=str, help='Root path of validation videos')
    parser.add_argument('--validation_labels', default='', type=str, help='Path to validation labels')
    parser.add_argument('--test_root', default='', type=str, help='Root path of test videos')
    parser.add_argument('--test_labels', default='', type=str, help='Path to test labels')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model_path', default='', type=str, help='path to trained model')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--video_type', default='rgb', type=str, help='rgb or flow')
    parser.add_argument('--classifier_type', default='video', type=str, help='video, multimodal, video_FT, video_gripper, FT_only, gripper_only, FTgripper_only')
    parser.add_argument('--robot_type_trainval', type=str, default='all', help='all, Toyota HSR or Kinova Gen3 for training and validation')
    parser.add_argument('--task_type_trainval', type=str, default='all', help='all, human to robot handover, robot to human handover for training and validation')
    parser.add_argument('--robot_type_test', type=str, default='all', help='all, Toyota HSR or Kinova Gen3 for testing')
    parser.add_argument('--task_type_test', type=str, default='all', help='all, human to robot handover, robot to human handover for testing')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor='val_loss',
        mode='min',
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    model = i3DTrainer(args, load_pretrained=True)

    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    main()

