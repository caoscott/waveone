import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data

from dataset import get_loader
# from evaluate import run_eval
from msssim import MSSSIM
from network import Decoder, Encoder
from train_options import parser

args = parser.parse_args()
print(args)

############### Data ###############
train_loader = get_loader(
    is_train=True,
    root=args.train, mv_dir=args.train_mv,
    args=args
)


def get_eval_loaders() -> Dict[str, data.DataLoader]:
  # We can extend this dict to evaluate on multiple datasets.
  eval_loaders = {
      'TVL': get_loader(
          is_train=False,
          root=args.eval, mv_dir=args.eval_mv,
          args=args),
  }
  return eval_loaders


############### Model ###############
encoder = Encoder(3, 128).cuda()
decoder = Decoder(128, 4).cuda()
binarizer = None
nets = [encoder, binarizer, decoder]

gpus = [int(gpu) for gpu in args.gpus.split(',')]
if len(gpus) > 1:
  print("Using GPUs {}.".format(gpus))
  net = nn.DataParallel(net, device_ids=gpus)

params = [{'params': net.parameters()} for net in nets]

solver = optim.Adam(
    params,
    lr=args.lr)

milestones = [int(s) for s in args.schedule.split(',')]
scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)
loss_fn = MSSSIM(val_range=2)

if not os.path.exists(args.model_dir):
  print("Creating directory %s." % args.model_dir)
  os.makedirs(args.model_dir)

############### Checkpoints ###############


def resume(index: int) -> None:
  names = ["encoder", "binarizer", "decoder"]

  for net_idx, net in enumerate(nets):
    if net is not None:
      name = names[net_idx]
      checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
          args.model_dir, args.save_model_name,
          name, index)

      print('Loading %s from %s...' % (name, checkpoint_path))
      net.load_state_dict(torch.load(checkpoint_path))


def save(index: int) -> None:
  names = ["encoder", "binarizer", "decoder"]

  for net_idx, net in enumerate(nets):
    if net is not None:
      torch.save(encoder.state_dict(),
                 '{}/{}_{}_{:08d}.pth'.format(
          args.model_dir, args.save_model_name,
          names[net_idx], index))


############### Training ###############

train_iter = 0
just_resumed = False
if args.load_model_name:
    print('Loading %s@iter %d' % (args.load_model_name,
                                  args.load_iter))

    resume(args.load_iter)
    train_iter = args.load_iter
    scheduler.last_epoch = train_iter - 1
    just_resumed = True


while True:

    for batch, (frame1, res, frame2, ctx_frames, _) in enumerate(train_loader):
        scheduler.step()
        train_iter += 1

        if train_iter > args.max_train_iters:
          break

        batch_t0 = time.time()

        solver.zero_grad()

        flows, residuals = decoder(encoder(torch.cat([frame1, frame2], dim=0)))

        bp_t0 = time.time()
        _, _, height, width = frame1.size()

        out_frame2 = F.grid_sample(frame1, flows) + residuals
        loss = loss_fn(frame2, out_frame2)

        bp_t1 = time.time()

        loss.backward()
        for net in nets:
            if net is not None:
                torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)

        solver.step()

        batch_t1 = time.time()

        print(
            '[TRAIN] Iter[{}]; LR: {}; Loss: {:.6f}; Backprop: {:.4f} sec; Batch: {:.4f} sec'.
            format(train_iter,
                   scheduler.get_lr()[0],
                   loss.item(),
                   bp_t1 - bp_t0,
                   batch_t1 - batch_t0))

        if train_iter % 100 == 0:
            print('Loss at each step:')
            print(('{:.4f} ' * args.iterations +
                   '\n').format(loss.item()))

        if train_iter % args.checkpoint_iters == 0:
            save(train_iter)

        if just_resumed or train_iter % args.eval_iters == 0 or train_iter == 100:
            print('Start evaluation...')

            # set_eval(nets)

            # eval_loaders = get_eval_loaders()
            # for eval_name, eval_loader in eval_loaders.items():
            #     eval_begin = time.time()
            #     eval_loss, mssim, psnr = run_eval(nets, eval_loader, args,
            #                                       output_suffix='iter%d' % train_iter)

            #     print('Evaluation @iter %d done in %d secs' % (
            #         train_iter, time.time() - eval_begin))
            #     print('%s Loss   : ' % eval_name
            #           + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
            #     print('%s MS-SSIM: ' % eval_name
            #           + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
            #     print('%s PSNR   : ' % eval_name
            #           + '\t'.join(['%.5f' % el for el in psnr.tolist()]))

            # set_train(nets)
            # just_resumed = False

    if train_iter > args.max_train_iters:
      print('Training done.')
      break
