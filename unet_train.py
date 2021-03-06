import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from waveone.dataset import get_master_loader
from waveone.losses import MSSSIM
from waveone.network import (CAE, AutoencoderUNet, Binarizer,
                             BitToContextDecoder, BitToFlowDecoder,
                             ContextToFlowDecoder, Encoder)
from waveone.network_parts import LambdaModule
from waveone.train_options import parser


def create_directories(dir_names: Tuple[str, ...]) -> None:
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


def save_tensor_as_img(
        t: torch.Tensor,
        name: str,
        args: argparse.Namespace,
        extension: str = "png",
) -> None:
    output_dir = os.path.join(args.out_dir, args.save_model_name)
    save_image(t + 0.5, os.path.join(output_dir, f"{name}.{extension}"))

############### Eval ###################


def eval_scores(
        frames1: List[torch.Tensor],
        frames2: List[torch.Tensor],
        prefix: str,
) -> Dict[str, torch.Tensor]:
    l1_loss_fn = nn.L1Loss(reduction="mean").cuda()
    msssim_fn = MSSSIM(val_range=1, normalize=True).cuda()

    assert len(frames1) == len(frames2)
    frame_len = len(frames1)
    msssim: torch.Tensor = 0.  # type: ignore
    l1: torch.Tensor = 0.  # type: ignore
    for frame1, frame2 in zip(frames1, frames2):
        l1 += l1_loss_fn(frame1, frame2)
        msssim += msssim_fn(frame1, frame2)
    return {f"{prefix}_l1": l1/frame_len,
            f"{prefix}_msssim": msssim/frame_len}


def forward_model(
        network: nn.Module, frame1: torch.Tensor, frame2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frames = torch.cat((frame1, frame2), dim=1)
    assert frame2.max() <= 0.5
    assert frame2.min() >= -0.5
    residuals = network(frames)
    reconstructed_frame2 = frame1 + residuals
    return residuals, reconstructed_frame2


def run_eval(
        eval_name: str,
        eval_loader: data.DataLoader,
        network: nn.Module,
        epoch: int,
        args: argparse.Namespace,
        writer: SummaryWriter,
        reuse_reconstructed: bool = True,
        fgsm: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    prefix = "" if reuse_reconstructed else "vcii_"
    network.eval()

    with torch.no_grad():
        eval_iterator = iter(eval_loader)
        frame1 = next(eval_iterator)[0]
        frames = [frame1]
        reconstructed_frames = []
        frame1 = frame1.cuda()

        for eval_iter, (frame2,) in enumerate(eval_iterator):
            frames.append(frame2)
            frame2 = frame2.cuda()
            _, reconstructed_frame2 = forward_model(network, frame1, frame2)
            reconstructed_frames.append(reconstructed_frame2.cpu())
            if args.save_out_img:
                save_tensor_as_img(
                    frame2, f"{prefix}_{eval_iter}_frame", args
                )
                save_tensor_as_img(
                    reconstructed_frame2,
                    f"{prefix}_{eval_iter}_reconstructed",
                    args
                )

            # Update frame1.
            if reuse_reconstructed:
                frame1 = reconstructed_frame2
            else:
                frame1 = frame2

        total_scores = {
            **eval_scores(frames[:-1], frames[1:], prefix + "eval_baseline"),
            **eval_scores(frames[1:], reconstructed_frames,
                          prefix + "eval_reconstructed"),
        }
        print(f"{eval_name} epoch {epoch}:")
        plot_scores(writer, total_scores, epoch)
        score_diffs = get_score_diffs(
            total_scores, ["reconstructed"], prefix + "eval")
        print_scores(score_diffs)
        plot_scores(writer, score_diffs, epoch)

        return total_scores, score_diffs


def plot_scores(
        writer: SummaryWriter,
        scores: Dict[str, torch.Tensor],
        train_iter: int
) -> None:
    for key, value in scores.items():
        writer.add_scalar(key, value, train_iter)


def get_score_diffs(
        scores: Dict[str, torch.Tensor],
        prefixes: List[str],
        prefix_type: str,
) -> Dict[str, torch.Tensor]:
    score_diffs = {}
    for score_type in ("msssim", "l1"):
        for prefix in prefixes:
            baseline_score = scores[f"{prefix_type}_baseline_{score_type}"]
            prefix_score = scores[f"{prefix_type}_{prefix}_{score_type}"]
            score_diffs[f"{prefix_type}_{prefix}_{score_type}_diff"
                        ] = prefix_score - baseline_score
    return score_diffs


def print_scores(scores: Dict[str, torch.Tensor]) -> None:
    for key, value in scores.items():
        print(f"{key}: {value.item() :.6f}")
    print("")


def train(args) -> List[nn.Module]:
    log_dir = os.path.join(args.log_dir, args.save_model_name)
    output_dir = os.path.join(args.out_dir, args.save_model_name)
    model_dir = os.path.join(args.model_dir, args.save_model_name)
    create_directories((output_dir, model_dir, log_dir))

    # logging.basicConfig(
    #     filename=os.path.join(log_dir, args.save_model_name + ".out"),
    #     filemode="w",
    #     level=logging.DEBUG,
    # )

    print(args)
    ############### Data ###############

    train_loader = get_master_loader(
        is_train=True,
        root=args.train,
        frame_len=4,
        sampling_range=12,
        args=args
    )
    eval_loader = get_master_loader(
        is_train=False,
        root=args.eval,
        frame_len=1,
        sampling_range=0,
        args=args,
    )
    writer = SummaryWriter(f"runs/{args.save_model_name}")

    ############### Model ###############
    network = AutoencoderUNet(6, shrink=1) if args.network == 'unet' \
        else CAE() if args.network == 'cae' \
        else LambdaModule(lambda x: x[:, 3:] - x[:, :3])
    network = network.cuda()
    nets: List[nn.Module] = [network]
    names = [args.network]
    solver = optim.Adam(
        network.parameters() if args.network != 'opt' else [torch.zeros((1,))],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    milestones = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=0.5)
    msssim_fn = MSSSIM(val_range=1, normalize=True).cuda()
    l1_loss_fn = nn.L1Loss(reduction="mean").cuda()
    l2_loss_fn = nn.MSELoss(reduction="mean").cuda()
    loss_fn = l2_loss_fn if args.loss == 'l2' else l1_loss_fn if args.loss == 'l1' \
        else LambdaModule(lambda a, b: -msssim_fn(a, b))

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

    ############### Training ###############

    train_iter = 0
    just_resumed = False
    if args.load_model_name:
        print(f'Loading {args.load_model_name}')
        resume()
        just_resumed = True

    def train_loop(
            frames: List[torch.Tensor],
    ) -> Iterator[Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        for net in nets:
            net.train()
        if args.network != 'opt':
            solver.zero_grad()

        reconstructed_frames = []
        reconstructed_frame2 = None

        loss: torch.Tensor = 0.  # type: ignore

        frame1 = frames[0].cuda()
        for frame2 in frames[1:]:
            frame2 = frame2.cuda()

            residuals, reconstructed_frame2 = forward_model(
                network, frame1, frame2)
            reconstructed_frames.append(reconstructed_frame2.cpu())
            loss += loss_fn(reconstructed_frame2, frame2)

            if args.save_max_l2:
                with torch.no_grad():
                    batch_l2 = ((frame2 - frame1 - residuals) ** 2).mean(
                        dim=-1).mean(dim=-1).mean(dim=-1).cpu()
                    max_batch_l2, max_batch_l2_idx = torch.max(batch_l2, dim=0)
                    max_batch_l2_frames = (
                        frame1[max_batch_l2_idx].cpu(),
                        frame2[max_batch_l2_idx].cpu(),
                        reconstructed_frame2[max_batch_l2_idx].detach().cpu(),
                    )
                    max_l2: float = max_batch_l2.item()  # type: ignore
                    yield max_l2, max_batch_l2_frames

            log_flow_context_residuals(writer, torch.abs(frame2 - frame1))

            frame1 = reconstructed_frame2.detach()

        scores = {
            **eval_scores(frames[:-1], frames[1:], "train_baseline"),
            **eval_scores(frames[1:], reconstructed_frames,
                          "train_reconstructed"),
        }

        if args.network != "opt":
            loss.backward()
            solver.step()

        writer.add_scalar("training_loss", loss.item(), train_iter)
        writer.add_scalar(
            "lr", solver.param_groups[0]["lr"], train_iter)  # type: ignore
        plot_scores(writer, scores, train_iter)
        score_diffs = get_score_diffs(scores, ["reconstructed"], "train")
        plot_scores(writer, score_diffs, train_iter)

    for epoch in range(args.max_train_epochs):
        for frames in train_loader:
            train_iter += 1
            max_epoch_l2, max_epoch_l2_frames = max(
                train_loop(frames), key=lambda x: x[0])

        if args.save_out_img:
            save_tensor_as_img(
                max_epoch_l2_frames[1],
                f"{max_epoch_l2 :.6f}_{epoch}_max_l2_frame",
                args,
            )
            save_tensor_as_img(
                max_epoch_l2_frames[2],
                f"{max_epoch_l2 :.6f}_{epoch}_max_l2_reconstructed",
                args,
            )

        if (epoch + 1) % args.checkpoint_epochs == 0:
            save()

        if just_resumed or ((epoch + 1) % args.eval_epochs == 0):
            run_eval("TVL", eval_loader, network,
                     epoch, args, writer, reuse_reconstructed=True)
            run_eval("TVL", eval_loader, network,
                     epoch, args, writer, reuse_reconstructed=False)
            scheduler.step()  # type: ignore
            just_resumed = False

    print('Training done.')
    logging.shutdown()
    return nets


if __name__ == '__main__':
    train(parser.parse_args())
