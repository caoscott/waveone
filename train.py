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
from network import (BitToContextDecoder, BitToFlowDecoder,
                     ContextToFlowDecoder, Encoder)
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
    context_vec_shape = (args.batch_size, 128,
                         args.patch or 144, args.patch or 176)
    encoder = Encoder(6, use_context=True).cuda()
    decoder = nn.Sequential(BitToContextDecoder(),
                            ContextToFlowDecoder(3)).cuda()
    # decoder = BitToFlowDecoder(3).cuda()
    nets = [encoder, decoder]

    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    if len(gpus) > 1:
        print("Using GPUs {}.".format(gpus))
        net = nn.DataParallel(net, device_ids=gpus)

    params = [{'params': net.parameters()} for net in nets]

    solver = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay)

    milestones = [int(s) for s in args.schedule.split(',')]
    scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)
    msssim_fn = MSSSIM(val_range=1, normalize=True).cuda()
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
            baseline_msssim_score = 0
            reconstructed_msssim_score = 0
            flow_msssim_score = 0
            context_vec = torch.zeros(context_vec_shape).cuda()

            for frame1, frame2, _, _ in eval_loader:
                batch_size = frame1.shape[0]
                frame1, frame2 = frame1.cuda(), frame2.cuda()
                # encoder_input = torch.cat([frame1, frame2], dim=1)
                flows, residuals, new_context_vec = decoder(
                    (encoder(frame1, frame2, context_vec), context_vec))
                context_vec = new_context_vec
                flow_frame2 = F.grid_sample(frame1, flows)
                reconstructed_frame2 = flow_frame2 + residuals
                baseline_msssim_score += msssim_fn(frame1, frame2) * batch_size
                reconstructed_msssim_score += msssim_fn(
                    frame2, reconstructed_frame2) * batch_size
                flow_msssim_score += msssim_fn(frame2,
                                               flow_frame2) * batch_size

            baseline_msssim_score /= len(eval_loader.dataset)
            # baseline_msssim_score_ctx /= len(eval_loader.dataset)
            reconstructed_msssim_score /= len(eval_loader.dataset)
            flow_msssim_score /= len(eval_loader.dataset)

            print(
                f"{eval_name} \t"
                f"Base MS-SSIM: {baseline_msssim_score: .6f} \t"
                # f"Base MS-SSIM CTX: {baseline_msssim_score_ctx: .6f} \t"
                f"Flow MS-SSIM: {flow_msssim_score: .6f} \t"
                f"Reconstructed MS-SSIM: {reconstructed_msssim_score: .6f}")

            writer.add_scalar(
                "base_msssim", baseline_msssim_score.item(), train_iter
            )
            writer.add_scalar(
                "flow_msssim", flow_msssim_score.item(), train_iter,
            )
            writer.add_scalar(
                "reconstructed_msssim", reconstructed_msssim_score.item(), train_iter
            )

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
        context_vec = torch.zeros(context_vec_shape, requires_grad=False).cuda()
        for frame1, frame2, _, _ in train_loader:
            frame1, frame2 = frame1.cuda(), frame2.cuda()
            train_iter += 1

            if train_iter > args.max_train_iters:
                break

            encoder.train()
            decoder.train()
            solver.zero_grad()

            # encoder_input = torch.cat([frame1, frame2], dim=1)
            flows, residuals, new_context_vec = decoder(
                (encoder(frame1, frame2, context_vec), context_vec))

            flow_frame2 = F.grid_sample(frame1, flows)
            reconstructed_frame2 = flow_frame2 + residuals
            loss = -msssim_fn(frame2, reconstructed_frame2) \
                + charbonnier_loss_fn(frame2, flow_frame2)

            loss.backward()
            for net in nets:
                if net is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

            solver.step()
            scheduler.step()

            context_vec = new_context_vec.detach()

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


with torch.autograd.set_detect_anomaly(True):
    train()
