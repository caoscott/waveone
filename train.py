import logging
import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from waveone.dataset import get_loader
from waveone.losses import MSSSIM, CharbonnierLoss
from waveone.network import (Binarizer, BitToContextDecoder, BitToFlowDecoder,
                             ContextToFlowDecoder, Encoder)
from waveone.train_options import parser


def create_directories(dir_names):
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            logging.info("Creating directory %s." % dir_name)
            os.makedirs(dir_name)


def train(args) -> List[nn.Module]:
    log_dir = os.path.join(args.log_dir, args.save_model_name)
    output_dir = os.path.join(args.out_dir, args.save_model_name)
    model_dir = os.path.join(args.model_dir, args.save_model_name)
    create_directories((output_dir, model_dir, log_dir))

    logging.basicConfig(filename=os.path.join(log_dir, args.save_model_name + ".out"))

    logging.info(args)
    ############### Data ###############

    train_loader = get_loader(
        is_train=True,
        root=args.train,
        frame_len=2,
        sampling_range=0,
        args=args
    )
    eval_loaders = {
        'TVL': get_loader(
            is_train=False,
            root=args.eval,
            frame_len=1,
            sampling_range=0,
            args=args,
        ),
    }
    writer = SummaryWriter()

    ############### Model ###############
    context_vec_train_shape = (args.batch_size, 512,
                               args.patch // 2 or 144, args.patch // 2 or 176)
    context_vec_test_shape = (args.eval_batch_size, 512, 144, 176)
    latent_vec_size = 512
    encoder = Encoder(6, latent_vec_size, use_context=False).cuda()
    # decoder = nn.Sequential(BitToContextDecoder(),
                            # ContextToFlowDecoder(3)).cuda()
    decoder = BitToFlowDecoder(args.bits, 3).cuda()
    binarizer = Binarizer(latent_vec_size, args.bits,
                          not args.binarize_off).cuda()
    nets = [encoder, binarizer, decoder]
    names = ["encoder", "binarizer", "decoder"]

    # gpus = [int(gpu) for gpu in args.gpus.split(',')]
    # if len(gpus) > 1:
    #     logging.info("Using GPUs {}.".format(gpus))
    #     net = nn.DataParallel(net, device_ids=gpus)

    params = [{'params': net.parameters()} for net in nets]

    solver = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    milestones = [int(s) for s in args.schedule.split(',')]
    # scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)
    msssim_fn = MSSSIM(val_range=1, normalize=True).cuda()
    # charbonnier_loss_fn = CharbonnierLoss().cuda()
    l1_loss_fn = nn.L1Loss(reduction="mean").cuda()

   ############### Checkpoints ###############

    def resume(index: int) -> None:
        for name, net in zip(names, nets):
            if net is not None:
                checkpoint_path = os.path.join(
                    args.model_dir, 
                    args.load_model_name, 
                    f"{name}.pth",
                )

                logging.info('Loading %s from %s...' % (name, checkpoint_path))
                net.load_state_dict(torch.load(checkpoint_path))

    def save(index: int) -> None:
        for name, net in zip(names, nets):
            if net is not None:
                checkpoint_path = os.path.join(
                    model_dir,
                    f'{name}.pth',
                )
                torch.save(net.state_dict(), checkpoint_path)

    def save_tensor_as_img(t: torch.Tensor, name: str, extension: str = "png") -> None:
        save_image(t + 0.5, os.path.join(output_dir, f"{name}.{extension}"))

    ############### Eval ###################
    def eval_scores(
        frames1: torch.Tensor,
        frames2: torch.Tensor,
        prefix: str
    ) -> Dict[str, torch.Tensor]:
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

    def get_score_diffs(scores, prefixes, prefix_type):
        score_diffs = {}
        for score_type in ("msssim", "l1"):
            for prefix in prefixes:
                baseline_score = scores[f"{prefix_type}_baseline_{score_type}"]
                prefix_score = scores[f"{prefix_type}_{prefix}_{score_type}"]
                score_diffs[f"{prefix_type}_{prefix}_{score_type}_diff"
                            ] = prefix_score - baseline_score
        return score_diffs

    def print_scores(scores):
        for key, value in scores.items():
            logging.info(f"{key}: {value.item() :.6f}")
        logging.info("")

    def add_dict(dict_a, dict_b):
        for key, value_b in dict_b.items():
            dict_a[key] += value_b
        return dict_a

    def log_flow_context_residuals(
        writer: SummaryWriter,
        flows: torch.Tensor,
        context_vec: torch.Tensor,
        residuals: torch.Tensor,
    ) -> None:
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
        writer.add_scalar("mean_input_residuals",
                          residuals.mean().item(), train_iter)
        writer.add_scalar("max_input_residuals",
                          residuals.max().item(), train_iter)
        writer.add_scalar("min_input_residuals",
                          residuals.min().item(), train_iter)

    def run_eval(
        eval_name: str,
        eval_loader: data.DataLoader,
        reuse_reconstructed: bool = True,
    ) -> None:
        for net in nets:
            net.eval()

        with torch.no_grad():
            context_vec = torch.zeros(context_vec_test_shape)  # .cuda()
            total_scores: Dict[str, float] = defaultdict(float)
            frame1 = None

            for eval_iter, (frame2,) in enumerate(eval_loader):
                frame2 = frame2.cuda()
                if frame1 is None:
                    frame1 = frame2
                    continue
                codes = binarizer(encoder(frame1, frame2, context_vec))
                flows, residuals, context_vec = decoder((codes, context_vec))
                flow_frame2 = F.grid_sample(frame1, flows)
                reconstructed_frame2 = (
                    flow_frame2 + residuals).clamp(-0.5, 0.5)

                prefix = "" if reuse_reconstructed else "vcii_"
                total_scores = add_dict(total_scores, eval_scores(
                    [frame1], [frame2], 
                    prefix + "eval_baseline",
                ))
                total_scores = add_dict(total_scores, eval_scores(
                    [frame2], [flow_frame2], 
                    prefix + "eval_flow",
                ))
                total_scores = add_dict(total_scores, eval_scores(
                    [frame2], [reconstructed_frame2], 
                    prefix + "eval_reconstructed",
                ))

                if args.save_out_img:
                    save_tensor_as_img(frame1, f"{prefix}{epoch}_{eval_iter}_frame1")
                    save_tensor_as_img(frame2, f"{prefix}{epoch}_{eval_iter}_frame2")
                    save_tensor_as_img(
                        reconstructed_frame2, 
                        f"{prefix}{epoch}_{eval_iter}_reconstructed_frame2"
                    )

                # Update frame1.
                if reuse_reconstructed:
                    frame1 = reconstructed_frame2
                else:
                    frame1 = frame2

            total_scores = {k: v/len(eval_loader.dataset)
                            for k, v in total_scores.items()}
            logging.info(f"{eval_name} epoch {epoch}:")
            plot_scores(writer, total_scores, epoch)
            score_diffs = get_score_diffs(
                total_scores, ("flow", "reconstructed"), prefix + "eval")
            print_scores(score_diffs)
            plot_scores(writer, score_diffs, epoch)

    ############### Training ###############

    train_iter = 0
    just_resumed = False
    if args.load_model_name:
        logging.info('Loading %s@iter %d' % (args.load_model_name,
                                      args.load_epoch))

        resume(args.load_epoch)
        # train_iter = args.load_epoch
        # scheduler.last_epoch = train_iter - 1
        just_resumed = True

    def train_loop(frames):
        for net in nets:
            net.train()
        solver.zero_grad()

        context_vec = torch.zeros(
            context_vec_train_shape, requires_grad=False)  # .cuda()
        flow_frames = []
        reconstructed_frames = []
        reconstructed_frame2 = None

        loss = 0.

        for frame1, frame2 in zip(frames[:-1], frames[1:]):
            # if reconstructed_frame2 is None:
                # frame1 = frame1.cuda()
            # else:
                # frame1 = reconstructed_frame2.detach()
            frame1 = frame1.cuda()
            del reconstructed_frame2

            # frame1, frame2 = frame1.cuda(), frame2.cuda()
            frame2 = frame2.cuda()
            # with 50% chance recycle old frame.
            # if reconstructed_frame2 is not None and random.randint(1, 2) == 1:

            codes = binarizer(encoder(frame1, frame2, context_vec))
            flows, residuals, context_vec = decoder((codes, context_vec))
            flow_frame2 = F.grid_sample(frame1, flows)
            flow_frames.append(flow_frame2.cpu())

            reconstructed_frame2 = (flow_frame2 + residuals).clamp(-0.5, 0.5)
            reconstructed_frames.append(reconstructed_frame2.cpu())

            batch_l1 = torch.abs(frame2-frame2-residuals).mean(
                dim=-1).mean(dim=-1).mean(dim=-1)
            batch_l1_cpu = batch_l1.detach().cpu()
            max_batch_l1, max_batch_l1_idx = torch.max(batch_l1_cpu, dim=0)
            min_batch_l1, min_batch_l1_idx = torch.min(batch_l1_cpu, dim=0)
            max_batch_l1_frames = (
                frame1[max_batch_l1_idx].cpu(),
                frame2[max_batch_l1_idx].cpu(),
                reconstructed_frame2[max_batch_l1_idx].detach().cpu(),
            )
            min_batch_l1_frames = (
                frame1[min_batch_l1_idx].cpu(),
                frame2[min_batch_l1_idx].cpu(),
                reconstructed_frame2[min_batch_l1_idx].detach().cpu(),
            )
            yield (
                max_batch_l1.item(), max_batch_l1_frames,
                min_batch_l1.item(), min_batch_l1_frames,
            )

            loss += batch_l1.mean()

            log_flow_context_residuals(
                writer, flows, context_vec, torch.abs(frame2 - frame1))

            del flows, residuals, flow_frame2

        scores = {
            **eval_scores(frames[:-1], frames[1:], "train_baseline"),
            **eval_scores(frames[1:], flow_frames, "train_flow"),
            **eval_scores(frames[1:], reconstructed_frames,
                          "train_reconstructed"),
        }

        # loss = -scores["train_reconstructed_msssim"] + scores["train_flow_l1"]
        # + charbonnier_loss_fn(frame2, flow_frame2)
        # loss = scores["train_reconstructed_l1"]
        loss.backward()
        # for net in nets:
        # if net is not None:
        # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

        solver.step()
        # scheduler.step()

        # context_vec = new_context_vec.detach()

        writer.add_scalar("training_loss", loss.item(), train_iter)
        plot_scores(writer, scores, train_iter)
        score_diffs = get_score_diffs(
            scores, ("flow", "reconstructed"), "train")
        plot_scores(writer, score_diffs, train_iter)

    for epoch in range(args.max_train_epochs):
        max_epoch_l1 = 0.
        min_epoch_l1 = float("inf")
        max_epoch_l1_frames = (None, None, None)
        min_epoch_l1_frames = (None, None, None)

        for frames in train_loader:
            train_iter += 1
            for (max_batch_l1, max_batch_l1_frames,
                 min_batch_l1, min_batch_l1_frames) in train_loop(frames):
                if max_epoch_l1 < max_batch_l1:
                    max_epoch_l1 = max_batch_l1
                    max_epoch_l1_frames = max_batch_l1_frames
                if min_epoch_l1 > min_batch_l1:
                    min_epoch_l1 = min_batch_l1
                    min_epoch_l1_frames = min_batch_l1_frames


        if args.save_out_img:
            for name, epoch_l1_frames, epoch_l1 in (
                    ("max_l1", max_epoch_l1_frames, max_epoch_l1),
                    ("min_l1", min_epoch_l1_frames, min_epoch_l1),
            ):
                save_tensor_as_img(
                    epoch_l1_frames[0],
                    f"{epoch_l1 :.6f}_{epoch}_{name}_frame1"
                )
                save_tensor_as_img(
                    epoch_l1_frames[1],
                    f"{epoch_l1 :.6f}_{epoch}_{name}_frame2"
                )
                save_tensor_as_img(
                    epoch_l1_frames[2],
                    f"{epoch_l1 :.6f}_{epoch}_{name}_reconstructed_frame2"
                )

        if (epoch + 1) % args.checkpoint_epochs == 0:
            save(epoch)

        if just_resumed or ((epoch + 1) % args.eval_epochs == 0):
            for eval_name, eval_loader in eval_loaders.items():
                run_eval(eval_name, eval_loader, reuse_reconstructed=True)
                run_eval(eval_name, eval_loader, reuse_reconstructed=False)
            just_resumed = False

    logging.info('Training done.')
    return nets


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
