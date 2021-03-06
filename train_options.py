"""Training options."""

import argparse

parser = argparse.ArgumentParser()

######## Data ########
parser.add_argument('--train', required=True, type=str,
                    help='Path to training data.')
parser.add_argument('--train-subset', required=True, type=str,
                    help='Path to subset of training data that is '
                         'properly compressed during evaluation.')
parser.add_argument('--eval', required=True, type=str,
                    help='Path to eval data.')

######## Model ########
parser.add_argument('--bits', default=64, type=int,
                    help='Bottle neck size.')
parser.add_argument('--binarize-off', action='store_true',
                    help='Turn off binarizer')
parser.add_argument('--patch', default=64, type=int,
                    help='Patch size.')
parser.add_argument('--train-type',
                    choices=("flow", "residual", "flow-residual"),
                    help='Choose a training type.')
parser.add_argument('--normalization', default='batch', type=str,
                    help='Set normalization in networks.')
parser.add_argument('--network',
                    choices=("unet", "opt", "cae",
                             "prednet", "small", "resnet", "resnet-ctx"),
                    help='Set network architecture.')

######## Learning ########
parser.add_argument('--max-train-epochs', type=int, default=20,
                    help='Max training epochs.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='Learning rate.')
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient clipping.')
parser.add_argument('--weight-decay', type=float,
                    default=1e-4, help='Weight decay')
# parser.add_argument('--schedule', default='100,200', type=str,
# help='Schedule milestones.')
# parser.add_argument('--gamma', type=float, default=0.5,
# help='LR decay factor.')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1,
                    help='Batch size for evaluation.')
parser.add_argument('--reconstructed-loss', choices=['l1', 'l2', 'msssim'],
                    help='Choose loss type for overall reconstruction.', default='msssim')
parser.add_argument('--flow-loss', choices=['l1', 'l2', 'msssim'],
                    help='Choose loss type for flow. No-op for --flow-off', default='l1')
parser.add_argument('--resblocks', type=int, default=8,
                    help='Number of resblocks to use in encoder and decoder.')
parser.add_argument('--iframe-iter', type=int, default=1000,
                    help='Set # of eval iterations for saving iframe.')
parser.add_argument('--sampling-range', type=int, default=0,
                    help='Number of frames in future to sample from for next frame during '
                         'training. 0 means picking the exact next frame in sequential order')
parser.add_argument('--frame-len', type=int, default=3,
                    help='Number of next frames to actually pick for training.')
parser.add_argument('--detach', action='store_true',
                    help='Detach gradients from previous frame when training from sequence of '
                         'frames. Used to make training more efficient.')
parser.add_argument('--lr-step-size', type=int, default=50,
                    help='Number of epochs after which to multiply existing learning rate by 0.1.')
parser.add_argument('--num-flows', type=int, default=1,
                    help='Number of optical flows.')

######## Experiment ########
parser.add_argument('--out-dir', type=str, default='output',
                    help='Output directory (for compressed codes & output images).')
parser.add_argument('--model-dir', type=str, default='model',
                    help='Path to model folder.')
parser.add_argument('--load-model', type=str,
                    help='Checkpoint name to load. (Do nothing if not specified.)')
# parser.add_argument('--load-epoch', type=int,
# help='Epoch of checkpoint to load.')
parser.add_argument('--save-model', type=str, default='demo',
                    help='Checkpoint name to save.')
# parser.add_argument('--save-codes', action='store_true',
# help='If true, write compressed codes during eval.')
parser.add_argument('--save-out-img', action='store_true',
                    help='If true, save output images during eval.')
parser.add_argument('--checkpoint-epochs', type=float, default=2,
                    help='Model checkpoint period.'
                    'If decimal, rounded to the nearest training iteration.')
parser.add_argument('--eval-epochs', type=float, default=10,
                    help='Evaluation period. '
                    'If decimal, rounded to the nearest training iteration.')
parser.add_argument('--mode', choices=("train, eval"), default="train",
                    help="Choose whether to train the model.")
# parser.add_argument('--plot-codes', action='store_true',
# help='If true, plot encoded bits from binarizer.')
