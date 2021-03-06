import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from waveone.dataset import get_loaders, get_master_loader
from waveone.losses import MSSSIM
from waveone.network import (CAE, AutoencoderUNet, Binarizer,
                             BitToContextDecoder, BitToFlowDecoder,
                             ContextToFlowDecoder, Encoder, SmallBinarizer,
                             SmallDecoder, SmallEncoder, UNet, WaveoneModel)
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
    save_image(t / 2 + 0.5, os.path.join(output_dir, f"{name}.{extension}"))

############### Eval ###################


def eval_scores(
        frames1: List[torch.Tensor],
        frames2: List[torch.Tensor],
        prefix: str,
) -> Dict[str, torch.Tensor]:
    l1_loss_fn = nn.L1Loss(reduction="mean")
    msssim_fn = MSSSIM(val_range=2, normalize=True)

    assert len(frames1) == len(frames2)
    frame_len = len(frames1)
    msssim: torch.Tensor = 0.  # type: ignore
    l1: torch.Tensor = 0.  # type: ignore
    for frame1, frame2 in zip(frames1, frames2):
        l1 += l1_loss_fn(frame1, frame2)
        msssim += msssim_fn(frame1, frame2)
    return {f"{prefix}_l1": l1/frame_len,
            f"{prefix}_msssim": msssim/frame_len}


def get_loss_fn(loss_type: str) -> nn.Module:
    assert loss_type in ["l1", "l2", "msssim"]
    return nn.MSELoss(reduction="mean") if loss_type == 'l2' \
        else nn.L1Loss(reduction="mean") if loss_type == 'l1' \
        else MSSSIM(val_range=2, normalize=True, negative=True)


def run_eval(
        eval_name: str,
        eval_loader: data.DataLoader,
        model: nn.Module,
        epoch: int,
        args: argparse.Namespace,
        writer: SummaryWriter,
        fgsm: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    model.eval()

    with torch.no_grad():
        eval_iterator = iter(eval_loader)
        frame1 = next(eval_iterator)[0]
        frames = [frame1]
        reconstructed_frames = []
        reconstructed_frames_vcii = []
        flow_frames = []
        flow_frames_vcii = []
        frame1 = torch.cat((frame1, frame1), dim=0).cuda()

        for eval_iter, (frame,) in enumerate(eval_iterator):
            frames.append(frame)
            frame = frame.cuda()
            frame2 = torch.cat((frame, frame), dim=0)
            assert frame1.shape == frame2.shape
            assert frame1.shape[0] == 2
            _, _, _, flow_frame, reconstructed_frame2 = model(
                frame1, frame2
            )
            reconstructed_frame2_cpu = reconstructed_frame2.cpu()
            flow_frame_cpu = flow_frame.cpu()

            reconstructed_frames.append(reconstructed_frame2_cpu[:1])
            reconstructed_frames_vcii.append(reconstructed_frame2_cpu[1:])
            flow_frames.append(flow_frame_cpu[:1])
            flow_frames_vcii.append(flow_frame_cpu[1:])
            if args.save_out_img:
                save_tensor_as_img(
                    frames[-1], f"{eval_name}_{eval_iter}_frame", args
                )
                save_tensor_as_img(
                    flow_frames[-1], f"{eval_name}_{eval_iter}_flow", args
                )
                save_tensor_as_img(
                    flow_frames_vcii[-1], f"{eval_name}_{eval_iter}_flow_vcii", args
                )
                save_tensor_as_img(
                    reconstructed_frames[-1],
                    f"{eval_name}_{eval_iter}_reconstructed",
                    args
                )
                save_tensor_as_img(
                    reconstructed_frames_vcii[-1],
                    f"{eval_name}_{eval_iter}_reconstructed_vcii",
                    args
                )

            # Update frame1.
            frame1 = torch.cat((reconstructed_frame2[0: 1], frame), dim=0)
            assert frame1.shape == frame2.shape
            assert frame1.shape[0] == 2

        total_scores: Dict[str, torch.Tensor] = {
            **eval_scores(frames[:-1], frames[1:], f"{eval_name}_baseline"),
            **eval_scores(frames[1:], flow_frames, f"{eval_name}_flow"),
            **eval_scores(frames[1:], reconstructed_frames,
                          f"{eval_name}_reconstructed"),
            # **eval_scores(frames[:-1], frames[1:], "vcii_eval_baseline"),
            **eval_scores(frames[1:], flow_frames_vcii, f"{eval_name}_vcii_flow"),
            **eval_scores(frames[1:], reconstructed_frames_vcii,
                          f"{eval_name}_vcii_reconstructed"),
        }

        print(f"{eval_name} epoch {epoch}:")
        plot_scores(writer, total_scores, epoch)
        print_scores(total_scores)
        score_diffs = get_score_diffs(
            total_scores,
            ["flow", "reconstructed", "vcii_flow", "vcii_reconstructed"],
            [eval_name]
        )
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
        prefix_types: List[str],
) -> Dict[str, torch.Tensor]:
    score_diffs = {}
    for prefix_type in prefix_types:
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


def resume(args: argparse.Namespace,
           model: nn.Module) -> None:
    checkpoint_path = os.path.join(
        args.model_dir,
        args.load_model_name,
        f"{args.network}.pth",
    )

    print('Loading %s from %s...' % (args.network, checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))


def save(args: argparse.Namespace,
         model: nn.Module) -> None:
    checkpoint_path = os.path.join(
        args.model_dir,
        args.save_model_name,
        f'{args.network}.pth',
    )
    torch.save(model.state_dict(), checkpoint_path)


def get_model(args: argparse.Namespace) -> nn.Module:
    if "waveone" in args.network:
        # context_vec_train_shape = (args.batch_size, 512,
                            #    args.patch // 2 or 144, args.patch // 2 or 176)
        # context_vec_test_shape = (args.eval_batch_size, 512, 144, 176)
        # unet = UNet(3, shrink=1)
        encoder = Encoder(6, args.bits, use_context=False)
        # decoder = nn.Sequential(BitToContextDecoder(),
        # ContextToFlowDecoder(3)).cuda()
        decoder = BitToFlowDecoder(args.bits, 3)
        binarizer = Binarizer(args.bits, args.bits,
                              not args.binarize_off)
        return WaveoneModel(encoder, binarizer, decoder, args.flow_off)
    if args.network == "cae":
        return CAE()
    if args.network == "unet":
        return AutoencoderUNet(6, shrink=1)
    if args.network == "opt":
        opt_encoder = LambdaModule(lambda f1, f2, _: f2 - f1)
        opt_binarizer = nn.Identity()  # type: ignore
        opt_decoder = LambdaModule(lambda t: (
            torch.tensor(0.), t[0], torch.tensor(0.)))
        return WaveoneModel(opt_encoder, opt_binarizer, opt_decoder, flow_off=True)
    if args.network == "small":
        small_encoder = SmallEncoder(6, args.bits)
        small_binarizer = SmallBinarizer(not args.binarize_off)
        small_decoder = SmallDecoder(args.bits, 3)
        return WaveoneModel(small_encoder, small_binarizer, small_decoder, args.flow_off)
    raise ValueError(f"No model type named {args.network}.")


def train(args) -> nn.Module:
    output_dir = os.path.join(args.out_dir, args.save_model_name)
    model_dir = os.path.join(args.model_dir, args.save_model_name)
    create_directories((output_dir, model_dir))

    print(args)
    ############### Data ###############

    train_loader = list(get_loaders(
        is_train=True,
        root=args.train,
        frame_len=args.frame_len,
        sampling_range=args.sampling_range,
        args=args,
    ).values())[0]
    train_sequential_loader = list(get_loaders(
        is_train=False,
        root=args.train,
        frame_len=1,
        sampling_range=0,
        args=args,
    ).values())[0]
    eval_loader = get_master_loader(
        is_train=False,
        root=args.eval,
        frame_len=1,
        sampling_range=0,
        args=args,
    )
    writer = SummaryWriter(f"runs/{args.save_model_name}", purge_step=0)

    ############### Model ###############
    model = get_model(args).cuda()
    solver = optim.Adam(
        model.parameters() if args.network != "opt" else [torch.zeros((1,))],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    milestones = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=0.5)
    reconstructed_loss_fn = get_loss_fn(args.reconstructed_loss).cuda()
    flow_loss_fn = get_loss_fn(args.flow_loss).cuda()

    def log_flow_context_residuals(
            writer: SummaryWriter,
            flows: torch.Tensor,
            context_vec: torch.Tensor,
            residuals: torch.Tensor,
    ) -> None:
        if args.network != "opt" and args.flow_off is False:
            flows_mean = flows.mean(dim=0).mean(dim=0).mean(dim=0)
            flows_max = flows.max(dim=0).values.max(  # type: ignore
                dim=0).values.max(dim=0).values  # type: ignore
            flows_min = flows.min(dim=0).values.min(  # type: ignore
                dim=0).values.min(dim=0).values  # type: ignore
            writer.add_histogram(
                "mean_flow_x", flows_mean[0].item(), train_iter)
            writer.add_histogram(
                "mean_flow_y", flows_mean[1].item(), train_iter)
            writer.add_histogram(
                "max_flow_x", flows_max[0].item(), train_iter)
            writer.add_histogram(
                "max_flow_y", flows_max[1].item(), train_iter)
            writer.add_histogram(
                "min_flow_x", flows_min[0].item(), train_iter)
            writer.add_histogram(
                "min_flow_y", flows_min[1].item(), train_iter)

        if "ctx" in args.network:
            writer.add_histogram("mean_context_vec_norm",
                                 context_vec.mean().item(), train_iter)
            writer.add_histogram("max_context_vec_norm",
                                 context_vec.max().item(), train_iter)
            writer.add_histogram("min_context_vec_norm",
                                 context_vec.min().item(), train_iter)

        writer.add_histogram("mean_input_residuals",
                             residuals.mean().item(), train_iter)
        writer.add_histogram("max_input_residuals",
                             residuals.max().item(), train_iter)
        writer.add_histogram("min_input_residuals",
                             residuals.min().item(), train_iter)

    ############### Training ###############

    train_iter = 0
    just_resumed = False
    if args.load_model_name:
        print(f'Loading {args.load_model_name}')
        resume(args, model)
        just_resumed = True

    def train_loop(
            frames: List[torch.Tensor],
    ) -> Iterator[Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        model.train()
        solver.zero_grad()

        context_vec = 0.  # .cuda()
        reconstructed_frames = []
        flow_frames = []
        reconstructed_frame2 = None
        loss: torch.Tensor = 0.  # type: ignore

        frame1 = frames[0].cuda()
        for frame2 in frames[1:]:
            frame2 = frame2.cuda()

            _, flows, residuals, flow_frame, reconstructed_frame2 = model(
                frame1, frame2
            )
            flow_frames.append(flow_frame.cpu())
            reconstructed_frames.append(reconstructed_frame2.cpu())
            loss += reconstructed_loss_fn(frame2, reconstructed_frame2) + (
                0 if args.flow_off else flow_loss_fn(frame2, flow_frame))

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
            else:
                yield 0, (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))

            log_flow_context_residuals(
                writer, flows, torch.tensor(context_vec), torch.abs(frame2 - frame1))

            frame1 = reconstructed_frame2  # .detach()
            # frame1 = frame2

        if args.network != "opt":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            solver.step()
        scores = {
            **eval_scores(frames[:-1], frames[1:], "train_baseline"),
            **eval_scores(frames[1:], flow_frames, "train_flow"),
            **eval_scores(frames[1:], reconstructed_frames,
                          "train_reconstructed"),
        }

        writer.add_scalar(
            "training_loss",
            loss.item() / (len(frames)-1),
            train_iter,
        )
        writer.add_scalar(
            "lr", solver.param_groups[0]["lr"], train_iter)  # type: ignore
        plot_scores(writer, scores, train_iter)
        score_diffs = get_score_diffs(scores, ["reconstructed"], ["train"])
        plot_scores(writer, score_diffs, train_iter)

    for epoch in range(args.max_train_epochs):
        for frames in train_loader:
            train_iter += 1
            max_epoch_l2, max_epoch_l2_frames = max(
                train_loop(frames), key=lambda x: x[0])

        if args.save_max_l2:
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
            save(args, model)

        if just_resumed or ((epoch + 1) % args.eval_epochs == 0):
            run_eval("eval", eval_loader, model,
                     epoch, args, writer)
            run_eval("training", train_sequential_loader, model,
                     epoch, args, writer)
            scheduler.step()  # type: ignore
            just_resumed = False

    print('Training done.')
    logging.shutdown()
    return model


if __name__ == '__main__':
    train(parser.parse_args())
