import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from waveone.dataset import get_loader
from waveone.losses import MSSSIM, CharbonnierLoss, TotalVariation, msssim
from waveone.network import (CAE, AutoencoderUNet, ResNetDecoder,
                             ResNetEncoder, SmallBinarizer, SmallDecoder,
                             SmallEncoder, UNet, WaveoneModel)
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
    output_dir = os.path.join(args.out_dir, args.save_model)
    save_image(t / 2 + 0.5, os.path.join(output_dir, f"{name}.{extension}"))

############### Eval ###################


def eval_scores(
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        name: str,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    assert frame1.shape == frame2.shape, (
        "frame1.shape {frame1.shape} != frame2.shape {frame2.shape}"
    )
    frame1 = frame1.reshape(-1,
                            frame1.shape[-3], frame1.shape[-2], frame1.shape[-1])
    frame2 = frame2.reshape(-1,
                            frame2.shape[-3], frame2.shape[-2], frame2.shape[-1])
    scores[f"{name}_l1"] = F.l1_loss(frame1, frame2, reduction="mean").item()
    scores[f"{name}_msssim"] = msssim(
        frame1, frame2, val_range=2, normalize=True,
    ).item()
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


def log_context_vec(context_vec: torch.Tensor, writer: SummaryWriter, epoch: int) -> None:
    writer.add_histogram(
        "context_vec_l1_norm_diff",
        context_vec[-1].abs().max() - context_vec[0].abs().max(),
        epoch,
    )


def run_eval(  # type: ignore
        eval_name: str,
        eval_loader: data.DataLoader,
        model: nn.Module,
        epoch: int,
        train_iter: int,
        args: argparse.Namespace,
        writer: SummaryWriter,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        score_collector: DefaultDict[
            str, List[float]] = defaultdict(list)
        for eval_idx, (frame_list, mask_list) in enumerate(eval_loader):
            frames = torch.stack(frame_list)
            masks = torch.stack(mask_list[1:])
            model_out = model(
                frames, iframe_iter=args.iframe_iter,
                reuse_frame=True, detach=False, collect_output=True,
            )
            for batch_i in range(model_out["reconstructed_frame2"].shape[1]):
                for seq_i in range(model_out["reconstructed_frame2"].shape[0]):
                    if masks[seq_i, batch_i].item() == 0:
                        break
                    save_frames = torch.stack([
                        frames[seq_i+1, batch_i],
                        model_out["flow_frame2"][seq_i, batch_i],
                        model_out["reconstructed_frame2"][seq_i, batch_i],
                    ])
                    vid_i = batch_i + eval_idx * args.eval_batch_size
                    save_tensor_as_img(
                        save_frames, f"{eval_name}_{vid_i}_{seq_i}", args)
            scores = {
                **eval_scores(
                    frames[1:],
                    torch.stack([frames[0]] *
                                (frames.shape[0]-1)),
                    f"{eval_name}_same_frame"
                ),
                **eval_scores(
                    frames[1:], frames[:-1],
                    f"{eval_name}_previous_frame"
                ),
                **eval_scores(
                    frames[1:], model_out["flow_frame2"],
                    f"{eval_name}_flow"
                ),
                **eval_scores(
                    frames[1:], model_out["reconstructed_frame2"],
                    f"{eval_name}_reconstructed"
                ),
            }
            for key, score in scores.items():
                score_collector[key].append(score * frame_list[0].shape[0])
            log_context_vec(model_out["context_vec"], writer, train_iter)
        total_scores: Dict[str, float] = {
            key: sum(score_list) / len(eval_loader.dataset)
            for key, score_list in score_collector.items()
        }

        print(f"{eval_name} epoch {epoch} iteration {train_iter}:")
        plot_scores(writer, total_scores, train_iter)
        print_scores(total_scores)
        return total_scores


def plot_scores(
        writer: SummaryWriter,
        scores: Dict[str, float],
        train_iter: int
) -> None:
    for key, value in scores.items():
        writer.add_scalar(key, value, train_iter)


def print_scores(scores: Dict[str, float]) -> None:
    for key, value in scores.items():
        print(f"{key}: {value :.6f}")
    print("")


def resume(args: argparse.Namespace,
           model: nn.Module) -> None:
    checkpoint_path = os.path.join(
        args.model_dir,
        args.load_model,
        f"{args.network}.pth",
    )

    print(f'Loading {args.network} from {checkpoint_path}...')
    model.load_state_dict(torch.load(checkpoint_path))


def save(args: argparse.Namespace,
         model: nn.Module) -> None:
    checkpoint_path = os.path.join(
        args.model_dir,
        args.save_model,
        f'{args.network}.pth',
    )
    torch.save(model.state_dict(), checkpoint_path)


def get_model(args: argparse.Namespace) -> nn.Module:
    # if "waveone" in args.network:
    #     # context_vec_train_shape = (args.batch_size, 512,
    #                         #    args.patch // 2 or 144, args.patch // 2 or 176)
    #     # context_vec_test_shape = (args.eval_batch_size, 512, 144, 176)
    #     # unet = UNet(3, shrink=1)
    #     encoder = Encoder(6, args.bits, use_context=False)
    #     # decoder = nn.Sequential(BitToContextDecoder(),
    #     # ContextToFlowDecoder(3)).cuda()
    #     decoder = BitToFlowDecoder(args.bits, 3)
    #     binarizer = Binarizer(args.bits, args.bits,
    #                           not args.binarize_off)
    #     return WaveoneModel(encoder, binarizer, decoder, args.train_type)
    flow_loss_fn = get_loss_fn(args.flow_loss).cuda()
    reconstructed_loss_fn = get_loss_fn(args.reconstructed_loss).cuda()
    if args.network == "cae":
        return CAE()
    if args.network == "unet":
        return AutoencoderUNet(6, shrink=1)
    if args.network == "opt":
        opt_encoder = LambdaModule(lambda f1, f2, _: f2 - f1)
        opt_binarizer = nn.Identity()  # type: ignore
        opt_decoder = LambdaModule(lambda t: {
            "flow": torch.zeros(1),
            "flow_grid": torch.zeros(1),
            "residuals": t[0],
            "context_vec": torch.zeros(1),
            "loss": torch.tensor(0.),
        })
        opt_decoder.num_flows = 1  # type: ignore
        return WaveoneModel(
            opt_encoder, opt_binarizer, opt_decoder, "residual",
            False, flow_loss_fn, reconstructed_loss_fn,
        )
    if args.network == "small":
        small_encoder = SmallEncoder(6, args.bits)
        small_binarizer = SmallBinarizer(not args.binarize_off)
        small_decoder = SmallDecoder(args.bits, 3)
        return WaveoneModel(
            small_encoder, small_binarizer, small_decoder, args.train_type,
            False, flow_loss_fn, reconstructed_loss_fn,
        )
    if "resnet" in args.network:
        use_context = "ctx" in args.network
        resnet_encoder = ResNetEncoder(
            6, args.bits, resblocks=args.resblocks, use_context=use_context)
        resnet_binarizer = SmallBinarizer(not args.binarize_off)
        resnet_decoder = ResNetDecoder(
            args.bits, 3, resblocks=args.resblocks, use_context=use_context,
            num_flows=args.num_flows)
        return WaveoneModel(
            resnet_encoder, resnet_binarizer, resnet_decoder, args.train_type,
            use_context, flow_loss_fn, reconstructed_loss_fn,
        )
    raise ValueError(f"No model type named {args.network}.")


def train(args) -> nn.Module:
    output_dir = os.path.join(args.out_dir, args.save_model)
    model_dir = os.path.join(args.model_dir, args.save_model)
    create_directories((output_dir, model_dir))

    print(args)
    ############### Data ###############

    train_paths = glob.glob(args.train)
    assert train_paths
    train_subset_paths = glob.glob(args.train_subset)
    assert train_subset_paths
    eval_paths = glob.glob(args.eval)
    assert eval_paths
    train_loader = get_loader(train_paths, is_train=True, args=args)
    train_subset_loader = get_loader(
        train_subset_paths, is_train=False, args=args)
    eval_loader = get_loader(eval_paths, is_train=False, args=args)

    writer = SummaryWriter(f"runs/{args.save_model}", purge_step=0)

    ############### Model ###############
    model = get_model(args)
    print(model)
    model = model.cuda()
    solver = optim.Adam(
        model.parameters() if args.network != "opt" else [torch.zeros((1,))],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = LS.StepLR(solver, step_size=args.lr_step_size, gamma=0.1)
    # tv = TotalVariation().cuda()

    def log_flow(
            writer: SummaryWriter,
            flows: torch.Tensor,
    ) -> None:
        if args.network != "opt" and "flow" in args.train_type:
            flows_x = flows[:, :, :, 0]
            flows_y = flows[:, :, :, 1]
            writer.add_histogram(
                "mean_flow_x", flows_x.mean().item(), train_iter)
            writer.add_histogram(
                "mean_flow_y", flows_y.mean().item(), train_iter)
            writer.add_histogram(
                "max_flow_x", flows_x.max().item(), train_iter)
            writer.add_histogram(
                "max_flow_y", flows_y.max().item(), train_iter)
            writer.add_histogram(
                "min_flow_x", flows_x.min().item(), train_iter)
            writer.add_histogram(
                "min_flow_y", flows_y.min().item(), train_iter)

    ############### Training ###############

    train_iter = 0
    assert args.checkpoint_epochs > 0, f"{args.checkpoint_epochs} <= 0"
    assert args.eval_epochs > 0, f"{args.eval_epochs} <= 0"
    checkpoint_iters = int(
        math.ceil(args.checkpoint_epochs * len(train_loader)))
    print(f"Saving model every {checkpoint_iters} training iterations.")
    eval_iters = int(math.ceil(args.eval_epochs * len(train_loader)))
    print(f"Evaluating every {eval_iters} training iterations.")

    if args.load_model:
        print(f'Loading {args.load_model}')
        resume(args, model)

    def train_loop(frame_list: List[torch.Tensor], log_iter: int) -> None:
        if np.random.random() < 0.5:
            frame_list = frame_list[::-1]
        frames = torch.stack(frame_list)

        model.train()
        solver.zero_grad()

        # context_vec = 0.  # .cuda()
        model_out = model(
            frames, iframe_iter=sys.maxsize, reuse_frame=True, detach=args.detach,
            collect_output=train_iter % log_iter == 0,
        )
        # frame1 = model_out["reconstructed_frame"]
        # if args.detach:
        #     frame1 = frame1.detach()

        if args.network != "opt":
            model_out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            solver.step()

        if args.network == "opt" or train_iter % log_iter == 0:
            scores = {
                **eval_scores(frames[1:],
                              torch.stack([frames[0]] * (frames.shape[0]-1)),
                              "train_same_frame"),
                **eval_scores(frames[1:], frames[:-1], "train_previous_frame"),
                **eval_scores(frames[1:], model_out["flow_frame2"], "train_flow"),
                **eval_scores(frames[1:], model_out["reconstructed_frame2"],
                              "train_reconstructed"),
            }
            writer.add_scalar(
                "training_loss", model_out["loss"].item(), train_iter,
            )
            writer.add_scalar(
                "lr", scheduler.get_lr()[0], train_iter)  # type: ignore
            plot_scores(writer, scores, train_iter)
            log_flow(writer, model_out["flow"])

    if args.load_model or args.mode == "eval":
        run_eval("eval", eval_loader, model, 0, 0, args, writer)
        run_eval("training", train_subset_loader,
                 model, 0, 0, args, writer)

    if args.mode == "train":
        for epoch in range(args.max_train_epochs):
            for frames, _ in train_loader:
                train_iter += 1
                train_loop(frames, log_iter=max(len(train_loader)//5, 1))

                if (train_iter + 1) % checkpoint_iters == 0:
                    save(args, model)
                if (train_iter + 1) % args.eval_epochs == 0:
                    run_eval("eval", eval_loader, model,
                             epoch, train_iter, args, writer)
                    run_eval("training", train_subset_loader,
                             model, epoch, train_iter, args, writer)

            scheduler.step()  # type: ignore
        print('Training done.')

    return model


if __name__ == '__main__':
    train(parser.parse_args())
