import pytorch_lightning as pl
from argparse import ArgumentParser
import trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
import torch
import pdb


def main():
    parser = ArgumentParser()

    parser.add_argument('--train_root', default='', type=str, help='Root path of training set')
    parser.add_argument('--validation_root', default='', type=str, help='Root path of val set')
    parser.add_argument('--test_root', default='', type=str, help='Root path of test set')
    parser.add_argument('--train_labels', default='', type=str, help='Root path of training labels')
    parser.add_argument('--validation_labels', default='', type=str, help='Root path of val labels')
    parser.add_argument('--test_labels', default='', type=str, help='Root path of test labels')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--checkpoint', default='', type=str, help='load trained weights')

    parser = trainer.SegmentationTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor='val_loss',
        every_n_epochs=1,
        mode='min',
        save_weights_only=True
    )
    trainer_obj = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    model = trainer.SegmentationTrainer(args)
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    trainer_obj.fit(model)
    trainer_obj.test(model)


if __name__ == '__main__':
    main()
