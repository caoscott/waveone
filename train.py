import os
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from dataset import get_loader
from losses import MSSSIM, CharbonnierLoss
from network import Decoder, Encoder
from train_options import parser


def train():
    args = parser.parse_args()
    print(args)

    ############### Data ###############
    def get_eval_loaders() -> Dict[str, data.DataLoader]:
        # We can extend this dict to evaluate on multiple datasets.
        eval_loaders = {
            'TVL': get_loader(
                is_train=False,
                root=args.eval, mv_dir=args.eval_mv,
                args=args),
        }
        return eval_loaders

    train_loader = get_loader(
        is_train=True,
        root=args.train, mv_dir=args.train_mv,
        args=args
    )
    writer = SummaryWriter()

    ############### Model ###############
    encoder = Encoder(6, 128).cuda()
    decoder = Decoder(128, 3).cuda()
    nets = [encoder, decoder]

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
    msssim_fn = MSSSIM(val_range=2, normalize=True).cuda()
    charbonnier_loss_fn = CharbonnierLoss().cuda()

    if not os.path.exists(args.model_dir):
        print("Creating directory %s." % args.model_dir)
        os.makedirs(args.model_dir)

   ############### Checkpoints ###############

    def resume(index: int) -> None:
        names = ["encoder", "decoder"]

        for net_idx, net in enumerate(nets):
            if net is not None:
                name = names[net_idx]
                checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
                    args.model_dir, args.save_model_name,
                    name, index)

                print('Loading %s from %s...' % (name, checkpoint_path))
                net.load_state_dict(torch.load(checkpoint_path))

    def save(index: int) -> None:
        names = ["encoder", "decoder"]

        for net_idx, net in enumerate(nets):
            if net is not None:
                torch.save(encoder.state_dict(),
                           '{}/{}_{}_{:08d}.pth'.format(
                    args.model_dir, args.save_model_name,
                    names[net_idx], index))

    ############### Eval ###################
    def run_eval(eval_name: str, eval_loader: data.DataLoader):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():

            reconstructed_msssim_score = 0
            flow_msssim_score = 0

            for frame1, _, frame2, _, _ in eval_loader:
                frame1, frame2 = frame1.cuda(), frame2.cuda()
                flows, residuals = decoder(
                    encoder(torch.cat([frame1, frame2], dim=1)))
                flow_frame2 = F.grid_sample(frame1, flows)
                reconstructed_frame2 = flow_frame2 + residuals
                reconstructed_msssim_score += msssim_fn(
                    frame2, reconstructed_frame2) * frame1.shape[0]
                flow_msssim_score += msssim_fn(frame2,
                                               flow_frame2) * frame1.shape[0]

            reconstructed_msssim_score /= len(eval_loader.dataset)
            flow_msssim_score /= len(eval_loader.dataset)

            print(
                f"{eval_name}"
                f"Flow MS-SSIM: {flow_msssim_score: .6f}\t"
                f"Reconstructed MS-SSIM: {reconstructed_msssim_score: .6f}\t")

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

        for _, (frame1, _, frame2, _, _) in enumerate(train_loader):
            frame1, frame2 = frame1.cuda(), frame2.cuda()
            train_iter += 1

            if train_iter > args.max_train_iters:
                break

            encoder.train()
            decoder.train()
            solver.zero_grad()

            batch_t0 = time.time()

            encoder_input = torch.cat([frame1, frame2], dim=1)
            flows, residuals = decoder(encoder(encoder_input))

            bp_t0 = time.time()

            flow_frame2 = F.grid_sample(frame1, flows)
            reconstructed_frame2 = flow_frame2 + residuals
            loss = -msssim_fn(frame2, reconstructed_frame2) + \
                charbonnier_loss_fn(frame2, flow_frame2)

            bp_t1 = time.time()

            loss.backward()
            for net in nets:
                if net is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

            solver.step()
            scheduler.step()

            batch_t1 = time.time()

            print(
                "[TRAIN] Iter[{}]; LR: {}; Loss: {:.6f}; "
                "Backprop: {:.4f} sec; Batch: {:.4f} sec".
                format(train_iter,
                       scheduler.get_lr()[0],
                       loss.item(),
                       bp_t1 - bp_t0,
                       batch_t1 - batch_t0),
                end="\r")
            writer.add_scalar("training_loss", loss.item(), train_iter)

            if train_iter % args.checkpoint_iters == 0:
                save(train_iter)

            if just_resumed or train_iter % args.eval_iters == 0:
                eval_loaders = get_eval_loaders()
                for eval_name, eval_loader in eval_loaders.items():
                    run_eval(eval_name, eval_loader)
                just_resumed = False

        if train_iter > args.max_train_iters:
            print('Training done.')
            break


train()
