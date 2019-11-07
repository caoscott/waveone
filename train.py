import os
# import time
from collections import defaultdict
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

    train_loader = get_loader(
        is_train=True,
        root=args.train, mv_dir=args.train_mv,
        args=args
    )
    eval_loaders = {
        'TVL': get_loader(
            is_train=False,
            root=args.eval, mv_dir=args.eval_mv,
            args=args),
    }
    writer = SummaryWriter()

    ############### Model ###############
    context_vec_train_shape = (args.batch_size, 512,
                               args.patch // 2 or 144, args.patch // 2 or 176)
    context_vec_test_shape = (args.eval_batch_size, 512, 144, 176)
    encoder = Encoder(6, use_context=False).cuda()
    # decoder = nn.Sequential(BitToContextDecoder(),
    #                         ContextToFlowDecoder(3)).cuda()
    decoder = BitToFlowDecoder(3).cuda()
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
    # charbonnier_loss_fn = CharbonnierLoss().cuda()
    l1_loss_fn = nn.L1Loss(reduction="mean").cuda()

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
    def eval_scores(frames1, frames2, prefix):
        assert len(frames1) == len(frames2)
        frame_len = len(frames1)
        msssim = 0.
        l1 = 0.
        for frame1, frame2 in zip(frames1, frames2):
            l1 += l1_loss_fn(frame1, frame2)
            msssim += msssim_fn(frame1, frame2)
        return {f"{prefix}_l1": l1/frame_len,
                f"{prefix}_msssim": msssim/frame_len}

    def plot_scores(writer, scores, train_iter):
        for key, value in scores.items():
            writer.add_scalar(key, value, train_iter)

    def print_scores(scores):
        for key, value in scores.items():
            print(f"{key}: {value.item(): .6f}")
        print()

    def add_dict(dict_a, dict_b):
        for key, value_b in dict_b.items():
            dict_a[key] += value_b
        return dict_a

    def log_flow_and_context(writer, flows, context_vec):
        flows_mean = flows.mean(dim=0).mean(dim=0).mean(dim=0)
        flows_max = flows.max(dim=0).values.max(dim=0).values.max(dim=0).values
        flows_min = flows.min(dim=0).values.min(dim=0).values.min(dim=0).values
        writer.add_scalar("mean_context_vec_norm",
                          context_vec.mean().item(), train_iter)
        writer.add_scalar("max_context_vec_norm",
                          context_vec.max().item(), train_iter)
        writer.add_scalar("min_context_vec_norm",
                          context_vec.min().item(), train_iter)
        writer.add_scalar(
            "mean_flow_x", flows_mean[0].item(), train_iter)
        writer.add_scalar(
            "mean_flow_y", flows_mean[1].item(), train_iter)
        writer.add_scalar(
            "max_flow_x", flows_max[0].item(), train_iter)
        writer.add_scalar(
            "max_flow_y", flows_max[1].item(), train_iter)
        writer.add_scalar(
            "min_flow_x", flows_min[0].item(), train_iter)
        writer.add_scalar(
            "min_flow_y", flows_min[1].item(), train_iter)

    def run_eval(eval_name: str, eval_loader: data.DataLoader):
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            context_vec = torch.zeros(context_vec_test_shape).cuda()
            total_scores = defaultdict(float)

            for frame1, frame2 in eval_loader:
                frame1, frame2 = frame1.cuda(), frame2.cuda()
                flows, residuals, context_vec = decoder(
                    (encoder(frame1, frame2, context_vec), context_vec))
                flow_frame2 = F.grid_sample(frame1, flows)
                reconstructed_frame2 = (flow_frame2 + residuals).clamp(0., 1.)

                scores = eval_scores([frame1], [frame2], "baseline") \
                    + eval_scores([frame2], [flow_frame2], "flow") \
                    + eval_scores([frame2], [reconstructed_frame2],
                                  "reconstructed")
                total_scores = add_dict(total_scores, scores)

            total_scores = {k: v/len(eval_loader.dataset)
                            for k, v in total_scores.items()}
            print(f"{eval_name}:")
            print_scores(total_scores)

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

    def train_loop(frames):
        encoder.train()
        decoder.train()
        solver.zero_grad()

        frames = [frame.cuda() for frame in frames]
        context_vec = torch.zeros(
            context_vec_train_shape, requires_grad=False).cuda()
        flow_frames = []
        reconstructed_frames = []

        for frame1, frame2 in zip(frames[:-1], frames[1:]):
            flows, residuals, context_vec = decoder(
                (encoder(frame1, frame2, context_vec), context_vec))

            flow_frame2 = F.grid_sample(frame1, flows)
            flow_frames.append(flow_frame2)

            reconstructed_frame2 = (flow_frame2 + residuals).clamp(0., 1.)
            reconstructed_frames.append(reconstructed_frame2)

            log_flow_and_context(writer, flows, context_vec)

        scores = \
            eval_scores(frames[:-1], frames[1:], "baseline") + \
            eval_scores(frames[1:], flow_frames, "flow") + \
            eval_scores(frames[1:], reconstructed_frames, "reconstructed")

        loss = -scores["reconstructed_msssim"] + scores["flow_l1"]
        # + charbonnier_loss_fn(frame2, flow_frame2)
        loss.backward()
        # for net in nets:
        # if net is not None:
        # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

        solver.step()
        scheduler.step()

        # context_vec = new_context_vec.detach()

        writer.add_scalar("training_loss", loss.item(), train_iter)
        plot_scores(writer, scores, train_iter)

    while True:
        for frames in train_loader:
            train_iter += 1
            if train_iter > args.max_train_iters:
                break

            train_loop(frames)

            if train_iter % args.checkpoint_iters == 0:
                save(train_iter)

            if just_resumed or train_iter % args.eval_iters == 0:
                for eval_name, eval_loader in eval_loaders.items():
                    run_eval(eval_name, eval_loader)
                just_resumed = False

        if train_iter > args.max_train_iters:
            print('Training done.')
            break


with torch.autograd.set_detect_anomaly(True):
    train()
