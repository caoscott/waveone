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
from torchvision import models
from torchvision.utils import save_image

from waveone.dataset import get_loader
from waveone.losses import MSSSIM, CharbonnierLoss
from waveone.network import (Binarizer, BitToContextDecoder, BitToFlowDecoder,
                             ContextToFlowDecoder, Encoder, UNet)
from waveone.train_options import parser


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super.__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def create_directories(dir_names):
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            print("Creating directory %s." % dir_name)
            os.makedirs(dir_name)


def add_dict(
    dict_a: Dict[str, torch.Tensor], dict_b: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    for key, value_b in dict_b.items():
        dict_a[key] += value_b
    return dict_a


def train(args) -> List[nn.Module]:
    log_dir = os.path.join(args.log_dir, args.save_model_name)
    output_dir = os.path.join(args.out_dir, args.save_model_name)
    model_dir = os.path.join(args.model_dir, args.save_model_name)
    create_directories((output_dir, model_dir, log_dir))

    logging.basicConfig(
        filename=os.path.join(log_dir, args.save_model_name + ".out"),
        filemode="w",
        level=logging.DEBUG,
    )

    print(args)
    ############### Data ###############

    train_loader = get_loader(
        is_train=True,
        root=args.train,
        frame_len=6,
        sampling_range=12,
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
    network = UNet(6, shrink=1).cuda()
    nets = [network]
    names = ["unet"]
    solver = optim.Adam(
        network.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LS.ReduceLROnPlateau(solver, mode="min", verbose=True)
    msssim_fn = MSSSIM(val_range=1, normalize=True).cuda()
    l1_loss_fn = nn.L1Loss(reduction="mean").cuda()
    l2_loss_fn = nn.MSELoss(reduction="mean").cuda()

   ############### Checkpoints ###############

    def resume() -> None:
        for name, net in zip(names, nets):
            if net is not None:
                checkpoint_path = os.path.join(
                    args.model_dir,
                    args.load_model_name,
                    f"{name}.pth",
                )

                print('Loading %s from %s...' % (name, checkpoint_path))
                net.load_state_dict(torch.load(checkpoint_path))

    def save() -> None:
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
        prefix: str,
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
            print(f"{key}: {value.item() :.6f}")
        print("")

    def log_flow_context_residuals(
        writer: SummaryWriter,
        residuals: torch.Tensor,
    ) -> None:
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
            total_scores: Dict[str, float] = defaultdict(float)
            frame1 = None

            for eval_iter, (frame2,) in enumerate(eval_loader):
                frame2 = frame2.cuda()
                if frame1 is None:
                    frame1 = frame2
                    continue
                residuals = network(frame2 - frame1)
                reconstructed_frame2 = frame2 + residuals

                prefix = "" if not reuse_reconstructed else "vcii_"
                total_scores = add_dict(total_scores, eval_scores(
                    [frame1], [frame2],
                    prefix + "eval_baseline",
                ))
                total_scores = add_dict(total_scores, eval_scores(
                    [frame2], [reconstructed_frame2],
                    prefix + "eval_reconstructed",
                ))

                if args.save_out_img:
                    save_tensor_as_img(
                        frame2, f"{prefix}{epoch}_{eval_iter}_frame")
                    save_tensor_as_img(
                        reconstructed_frame2,
                        f"{prefix}{epoch}_{eval_iter}_reconstructed"
                    )

                # Update frame1.
                if reuse_reconstructed:
                    frame1 = reconstructed_frame2
                else:
                    frame1 = frame2

            total_scores = {k: v/len(eval_loader.dataset)
                            for k, v in total_scores.items()}
            print(f"{eval_name} epoch {epoch}:")
            plot_scores(writer, total_scores, epoch)
            score_diffs = get_score_diffs(
                total_scores, ("reconstructed",), prefix + "eval")
            print_scores(score_diffs)
            plot_scores(writer, score_diffs, epoch)

    ############### Training ###############

    train_iter = 0
    just_resumed = False
    if args.load_model_name:
        print(f'Loading {args.load_model_name}')
        resume()
        just_resumed = True

    def train_loop(frames):
        for net in nets:
            net.train()
        solver.zero_grad()

        reconstructed_frames = []
        reconstructed_frame2 = None

        loss = 0.

        for frame1, frame2 in zip(frames[:-1], frames[1:]):
            if reconstructed_frame2 is None:
                frame1 = frame1.cuda()
            else:
                frame1 = reconstructed_frame2.detach()
            frame2 = frame2.cuda()

            residuals = network(torch.cat((frame1, frame2), dim=1))

            reconstructed_frame2 = (frame2 + residuals).clamp(-0.5, 0.5)
            reconstructed_frames.append(reconstructed_frame2.cpu())

            with torch.no_grad():
                batch_l2 = ((frame2 - frame1 - residuals) ** 2).mean(
                    dim=-1).mean(dim=-1).mean(dim=-1)
                batch_l2_cpu = batch_l2.detach().cpu()
                max_batch_l2, max_batch_l2_idx = torch.max(batch_l2_cpu, dim=0)
                min_batch_l2, min_batch_l2_idx = torch.min(batch_l2_cpu, dim=0)
                max_batch_l2_frames = (
                    frame1[max_batch_l2_idx].cpu(),
                    frame2[max_batch_l2_idx].cpu(),
                    reconstructed_frame2[max_batch_l2_idx].detach().cpu(),
                )
                min_batch_l2_frames = (
                    frame1[min_batch_l2_idx].cpu(),
                    frame2[min_batch_l2_idx].cpu(),
                    reconstructed_frame2[min_batch_l2_idx].detach().cpu(),
                )
                yield (
                    max_batch_l2.item(), max_batch_l2_frames,
                    min_batch_l2.item(), min_batch_l2_frames,
                )

            loss -= msssim_fn(frame1 + residuals, frame2)

            log_flow_context_residuals(writer, torch.abs(frame2 - frame1))

        scores = {
            **eval_scores(frames[:-1], frames[1:], "train_baseline"),
            **eval_scores(frames[1:], reconstructed_frames,
                          "train_reconstructed"),
        }

        loss.backward()
        solver.step()
        scheduler.step(loss.item())

        writer.add_scalar("training_loss", loss.item(), train_iter)
        writer.add_scalar("lr", scheduler.get_lr()[0], train_iter)
        plot_scores(writer, scores, train_iter)
        score_diffs = get_score_diffs(scores, ("reconstructed",), "train")
        plot_scores(writer, score_diffs, train_iter)

    for epoch in range(args.max_train_epochs):
        max_epoch_l1 = 0.
        min_epoch_l1 = float("inf")
        max_epoch_l1_frames = (None, None, None)
        min_epoch_l1_frames = (None, None, None)

        for frames in train_loader:
            train_iter += 1
            for (max_batch_l2, max_batch_l2_frames,
                 min_batch_l2, min_batch_l2_frames) in train_loop(frames):
                if max_epoch_l1 < max_batch_l2:
                    max_epoch_l1 = max_batch_l2
                    max_epoch_l1_frames = max_batch_l2_frames
                if min_epoch_l1 > min_batch_l2:
                    min_epoch_l1 = min_batch_l2
                    min_epoch_l1_frames = min_batch_l2_frames

        if args.save_out_img:
            for name, epoch_l1_frames, epoch_l1 in (
                    ("max_l1", max_epoch_l1_frames, max_epoch_l1),
                    ("min_l1", min_epoch_l1_frames, min_epoch_l1),
            ):
                save_tensor_as_img(
                    epoch_l1_frames[1],
                    f"{epoch_l1 :.6f}_{epoch}_{name}_frame"
                )
                save_tensor_as_img(
                    epoch_l1_frames[2],
                    f"{epoch_l1 :.6f}_{epoch}_{name}_reconstructed"
                )

        if (epoch + 1) % args.checkpoint_epochs == 0:
            save()

        if just_resumed or ((epoch + 1) % args.eval_epochs == 0):
            for eval_name, eval_loader in eval_loaders.items():
                run_eval(eval_name, eval_loader, reuse_reconstructed=True)
                run_eval(eval_name, eval_loader, reuse_reconstructed=False)
            just_resumed = False

    print('Training done.')
    logging.shutdown()
    return nets


if __name__ == '__main__':
    train(parser.parse_args())
