"""Training options."""

import argparse

parser = argparse.ArgumentParser()

######## Data ########
parser.add_argument('--train', required=True, type=str,
                    help='Path to training data.')
parser.add_argument('--eval', required=True, type=str,
                    help='Path to eval data.')

######## Model ########
parser.add_argument('--bits', default=256, type=int,
                    help='Bottle neck size.')
parser.add_argument('--binarize-off', action='store_true',
                    help='Turn off binarizer')
parser.add_argument('--patch', default=64, type=int,
                    help='Patch size.')
parser.add_argument('--flow-off', action='store_true',
                    help='Turn off flow')
parser.add_argument('--normalization', default='batch', type=str,
                    help='Set normalization in networks.')
parser.add_argument('--network', choices=("unet", "opt", "cae", "waveone", "waveone-ctx"),
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
parser.add_argument('--schedule', default='100,200', type=str,
                    help='Schedule milestones.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1,
                    help='Batch size for evaluation.')
parser.add_argument('--reconstructed-loss', choices=['l1', 'l2', 'msssim'],
                    help='Choose loss type for overall reconstruction.', default='msssim')
parser.add_argument('--flow-loss', choices=['l1', 'l2', 'msssim'],
                    help='Choose loss type for flow. No-op for --flow-off', default='l1')
parser.add_argument('--sampling-range', type=int, default=5,
                    help='Number of frames in future to sample from for next frame during '
                         'training. 0 means picking the exact next frame in sequential order')
parser.add_argument('--frame-len', type=int, default=3,
                    help='Number of next frames to actually pick for training.')

# To save computation, we compute objective for multiple
# crops for each forward pass.
parser.add_argument('--gpus', default='0', type=str,
                    help='GPU indices separated by comma, e.g. \"0,1\".')

######## Experiment ########
parser.add_argument('--out-dir', type=str, default='output',
                    help='Output directory (for compressed codes & output images).')
parser.add_argument('--model-dir', type=str, default='model',
                    help='Path to model folder.')
parser.add_argument('--load-model-name', type=str,
                    help='Checkpoint name to load. (Do nothing if not specified.)')
# parser.add_argument('--load-epoch', type=int,
# help='Epoch of checkpoint to load.')
parser.add_argument('--save-model-name', type=str, default='demo',
                    help='Checkpoint name to save.')
parser.add_argument('--save-codes', action='store_true',
                    help='If true, write compressed codes during eval.')
parser.add_argument('--save-out-img', action='store_true',
                    help='If true, save output images during eval.')
parser.add_argument('--checkpoint-epochs', type=int, default=20,
                    help='Model checkpoint period.')
parser.add_argument('--eval-epochs', type=int, default=10,
                    help='Evaluation period.')
parser.add_argument('--save-max-l2', action='store_true',
                    help='If true, save max l2 images for train epoch.')
