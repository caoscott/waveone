import argparse
import glob
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

from waveone.dataset import get_loaders
from waveone.losses import MSSSIM, CharbonnierLoss, TotalVariation, msssim
from waveone.network import (CAE, AutoencoderUNet, Binarizer,
                             BitToContextDecoder, BitToFlowDecoder,
                             ContextToFlowDecoder, Encoder, SmallBinarizer,
                             SmallDecoder, SmallEncoder, UNet, WaveoneModel)
from waveone.network_parts import LambdaModule
from waveone.train_options import parser


def create_directories(dir_names: Tuple[str, ...]) -> None:
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            print(f"Creating directory {dir_name}.")
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
        dict2: Dict[str, List[torch.Tensor]],
        name: str,
) -> Dict[str, torch.Tensor]:
    msssim_fn = MSSSIM(val_range=2, normalize=True)
    scores: Dict[str, torch.Tensor] = {}
    f1 = torch.cat(frames1, dim=0)

    for prefix, frames2 in dict2.items():
        f2 = torch.cat(frames2, dim=0)
        assert f1.shape == f2.shape
        scores[f"{name}_{prefix}l1"] = F.l1_loss(f1, f2, reduction="mean")
        scores[f"{name}_{prefix}msssim"] = msssim(
            f1, f2, val_range=2, normalize=True,
        )
    return scores


def get_loss_fn(loss_type: str) -> nn.Module:
    if loss_type == 'l2':
        return nn.MSELoss(reduction="mean")
    if loss_type == 'l1':
        return nn.L1Loss(reduction="mean")
    if loss_type == "msssim":
        return MSSSIM(val_range=2, normalize=True, negative=True)
    if loss_type == "charbonnier":
        return CharbonnierLoss()
    raise ValueError(f"{loss_type} is not an appropriate loss.")


def run_eval(
        eval_name: str,
        eval_loader: data.DataLoader,
        model: nn.Module,
        epoch: int,
        args: argparse.Namespace,
        writer: SummaryWriter,
) -> Dict[str, torch.Tensor]:
    model.eval()

    with torch.no_grad():
        eval_iterator = iter(eval_loader)
        frame1 = next(eval_iterator)[0]
        frames = [frame1]
        reconstructed_frames: Dict[str, List[torch.Tensor]] = defaultdict(list)
        flow_frames: Dict[str, List[torch.Tensor]] = defaultdict(list)
        frame1 = torch.cat((frame1, frame1, frame1), dim=0).cuda()

        for eval_iter, (frame,) in enumerate(eval_iterator):
            frames.append(frame)
            frame = frame.cuda()
            frame2 = torch.cat((frame, frame, frame), dim=0)
            assert frame1.shape == frame2.shape
            assert frame1.shape[0] == 3
            model_out = model(frame1, frame2)

            reconstructed_frame_cpu = model_out["reconstructed_frame"].cpu()
            flow_frame_cpu = model_out["flow_frame"].cpu()

            if args.save_out_img:
                for frame_i, prefix in enumerate(("", "vcii_", "iframe_")):
                    flow_i = flow_frame_cpu[frame_i].unsqueeze(0)
                    reconstructed_i = reconstructed_frame_cpu[frame_i].unsqueeze(
                        0)
                    flow_frames[prefix].append(flow_i)
                    reconstructed_frames[prefix].append(reconstructed_i)
                    save_tensor_as_img(
                        frames[-1], f"{prefix}{eval_name}_{eval_iter}_frame", args
                    )
                    save_tensor_as_img(
                        flow_i, f"{prefix}{eval_name}_{eval_iter}_flow", args
                    )
                    save_tensor_as_img(
                        reconstructed_i,
                        f"{prefix}{eval_name}_{eval_iter}_reconstructed",
                        args
                    )

            # Update frame1.
            frame1 = torch.cat(
                (
                    model_out["reconstructed_frame"][:1],
                    frame,
                    frame if (eval_iter+1) % args.iframe_iter == 0
                    else model_out["reconstructed_frame"][2:],
                ), dim=0
            )

        total_scores: Dict[str, torch.Tensor] = {
            **eval_scores(frames[:-1], {"": frames[1:]}, f"{eval_name}_baseline"),
            **eval_scores(frames[1:], flow_frames, f"{eval_name}_flow"),
            **eval_scores(frames[1:], reconstructed_frames, f"{eval_name}_reconstructed"),
        }

        print(f"{eval_name} epoch {epoch}:")
        plot_scores(writer, total_scores, epoch)
        print_scores(total_scores)
        return total_scores


def plot_scores(
        writer: SummaryWriter,
        scores: Dict[str, torch.Tensor],
        train_iter: int
) -> None:
    for key, value in scores.items():
        writer.add_scalar(key, value.item(), train_iter)


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

    print(f'Loading {args.network} from {checkpoint_path}...')
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
        return WaveoneModel(encoder, binarizer, decoder, args.train_type)
    if args.network == "cae":
        return CAE()
    if args.network == "unet":
        return AutoencoderUNet(6, shrink=1)
    if args.network == "opt":
        opt_encoder = LambdaModule(lambda f1, f2, _: f2 - f1)
        opt_binarizer = nn.Identity()  # type: ignore
        opt_decoder = LambdaModule(lambda t: {
            "flow": 0.,
            "flow_grid": 0.,
            "residuals": t[0],
            "context_vec": t[1],
        })
        return WaveoneModel(opt_encoder, opt_binarizer, opt_decoder, train_type="residual")
    if args.network == "small":
        small_encoder = SmallEncoder(6, args.bits)
        small_binarizer = SmallBinarizer(not args.binarize_off)
        small_decoder = SmallDecoder(args.bits, 3)
        return WaveoneModel(small_encoder, small_binarizer, small_decoder, args.train_type)
    raise ValueError(f"No model type named {args.network}.")


def train(args) -> nn.Module:
    output_dir = os.path.join(args.out_dir, args.save_model_name)
    model_dir = os.path.join(args.model_dir, args.save_model_name)
    create_directories((output_dir, model_dir))

    print(args)
    ############### Data ###############

    train_paths = glob.glob(os.path.join(args.train, '*.pkl'))
    assert train_paths
    train_subset_paths = glob.glob(os.path.join(args.train_subset, '*.pkl'))
    assert train_subset_paths
    eval_paths = glob.glob(os.path.join(args.eval, '*.pkl'))
    assert eval_paths

    writer = SummaryWriter(f"runs/{args.save_model_name}", purge_step=0)

    ############### Model ###############
    model = get_model(args).cuda()
    solver = optim.Adam(
        model.parameters() if args.network != "opt" else [torch.zeros((1,))],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LS.StepLR(solver, step_size=40, gamma=0.5)
    reconstructed_loss_fn = get_loss_fn(args.reconstructed_loss).cuda()
    flow_loss_fn = get_loss_fn(args.flow_loss).cuda()
    # tv = TotalVariation().cuda()

    def log_flow_context_residuals(
            writer: SummaryWriter,
            flows: torch.Tensor,
            context_vec: torch.Tensor,
            residuals: torch.Tensor,
    ) -> None:
        if args.network != "opt" and "flow" in args.train_type:
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

    def train_loop(frames: List[torch.Tensor]) -> None:
        if np.random.random() < 0.5:
            frames = frames[::-1]

        model.train()
        solver.zero_grad()

        context_vec = 0.  # .cuda()
        reconstructed_frames = []
        flow_frames = []
        loss: torch.Tensor = 0.  # type: ignore
        frames = [frame.cuda() for frame in frames]

        frame1 = frames[0]
        for frame2 in frames[1:]:
            model_out = model(frame1, frame2)
            if "flow" in args.train_type:
                loss += (
                    flow_loss_fn(frame2, model_out["flow_frame"])
                    #  + 0.01 * tv(model_out["flow"])
                )
            if "residual" in args.train_type:
                loss += reconstructed_loss_fn(frame2,
                                              model_out["reconstructed_frame"])

            flow_frames.append(model_out["flow_frame"])
            reconstructed_frames.append(
                model_out["reconstructed_frame"])

            log_flow_context_residuals(
                writer,
                model_out["flow"],
                torch.tensor(context_vec),
                torch.abs(frame2 - frame1)
            )

            frame1 = model_out["reconstructed_frame"]
            if args.detach:
                frame1 = frame1.detach()

        loss /= len(frames) - 1
        if args.network != "opt":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            solver.step()
        scores = {
            **eval_scores(frames[:-1], {"": frames[1:]}, "train_baseline"),
            **eval_scores(frames[1:], {"": flow_frames}, "train_flow"),
            **eval_scores(frames[1:], {"": reconstructed_frames},
                          "train_reconstructed"),
        }

        writer.add_scalar(
            "training_loss",
            loss.item(),
            train_iter,
        )
        writer.add_scalar(
            "lr", solver.param_groups[0]["lr"], train_iter)  # type: ignore
        plot_scores(writer, scores, train_iter)

    for epoch in range(args.max_train_epochs):
        for train_loader in get_loaders(train_paths, is_train=True, args=args):
            for frames in train_loader:
                train_iter += 1
                train_loop(frames)

        if (epoch + 1) % args.checkpoint_epochs == 0:
            save(args, model)

        if just_resumed or ((epoch + 1) % args.eval_epochs == 0):
            for eval_loader in get_loaders(eval_paths, is_train=False, args=args):
                run_eval("eval", eval_loader, model, epoch, args, writer)
            for train_subset_loader in get_loaders(
                train_subset_paths, is_train=False, args=args
            ):
                run_eval("training", train_subset_loader,
                         model, epoch, args, writer)
            scheduler.step()  # type: ignore
            just_resumed = False

    print('Training done.')
    return model


if __name__ == '__main__':
    train(parser.parse_args())
