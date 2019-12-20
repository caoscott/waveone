import argparse
import copy
import glob
import os
import random
import time
from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, List, Tuple

import cv2
import hickle as hkl
import numpy as np
import torch
import torch.utils.data as data


class ImageList(data.Dataset):
    def __init__(
        self,
        imgs: Tuple[np.ndarray, ...],
        is_train: bool,
        args: argparse.Namespace,
        frame_len: int,
        sampling_range: int,
    ) -> None:
        super().__init__()
        self.imgs = imgs
        self.is_train = is_train
        self.frame_len = frame_len
        self.padding_len = max(frame_len - len(imgs), 0)
        self.sampling_range = sampling_range
        self.args = args

        assert frame_len > 0
        assert sampling_range == 0 or sampling_range >= frame_len
        # assert len(self.imgs) >= self.frame_len

    def __getitem__(
            self, 
            index: int
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if self.padding_len > 0:
            images = self.imgs
        elif self.sampling_range > 0:
            idx_sampling_range = min(
                self.sampling_range, len(self.imgs)-index)
            offsets = np.random.permutation(idx_sampling_range)[
                :self.frame_len]
            images = tuple(self.imgs[index + offset]
                           for offset in np.sort(offsets))
        else:
            images = self.imgs[index: index+self.frame_len]

        if self.is_train:
            # images = contrast_cv2(brightness_cv2(flip_cv2(images)))
            images = flip_cv2(images)
            if self.args.patch:
                images = multi_crop_cv2(images, self.args.patch)
        images = tuple(square_cv2(img) for img in images)

        frames: Tuple[torch.Tensor, ...] = tuple(
            np_to_torch(img.astype(np.float64)/255*2 - 1) for img in images
        )
        existence_mask: Tuple[torch.Tensor, ...] = tuple(  # type: ignore
            torch.ones((1, 1, 1, 1))) * len(images)

        if self.padding_len > 0:
            frames += tuple(  # type: ignore
                torch.zeros_like(self.imgs[0])) * self.padding_len
            existence_mask += (tuple(  # type: ignore
                torch.zeros((1, 1, 1, 1))) * self.padding_len
            )

        if self.args.network == "opt":
            for frame in frames:
                assert frame.max() <= 1  # type: ignore
                assert frame.min() >= -1  # type: ignore
        assert len(frames) == self.frame_len
        assert len(frames) == len(existence_mask)

        return frames, existence_mask

    def __len__(self) -> int:
        if self.padding_len > 0:
            return 1
        return len(self.imgs) - self.frame_len + 1


class RandomVidSequenceSampler(data.Sampler):
    def __init__(self, dataset: data.Dataset, frame_len: int) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.frame_len = frame_len
        self.num_samples = len(
            dataset) // frame_len - 1 if len(dataset) % frame_len == 0 else len(
            dataset) // frame_len

    def __iter__(self) -> Iterator[int]:
        start = np.random.randint(self.frame_len + 1 if len(self.dataset) %
                                  self.frame_len == 0 else len(self.dataset) % self.frame_len + 1)
        indices = np.random.permutation(
            np.arange(self.num_samples) * self.frame_len + start)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


# def get_vid_id(filename: str) -> str:
#     return "_".join(filename.split("_")[:-1])


def get_loaders(
        paths: List[str],
        is_train: bool,
        args: argparse.Namespace,
) -> Iterator[data.DataLoader]:
    for hkl_path in paths:
        image_list = load_hkl_images(hkl_path)
        max_frame_len = max(len(images) for images in image_list)
        datasets = convert_images_to_datasets(
            image_list,
            is_train=is_train,
            args=args,
            frame_len=args.frame_len if is_train else max_frame_len,
            sampling_range=args.sampling_range if is_train else 0,
        )
        concat_dataset: data.Dataset = data.ConcatDataset(datasets)
        loader = data.DataLoader(
            concat_dataset,
            batch_size=args.batch_size,
            sampler=RandomVidSequenceSampler(concat_dataset, args.frame_len),
            num_workers=4,
            drop_last=True,
        ) if is_train else data.DataLoader(
            concat_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )
        print('Loader for {} sequences ({} batches) created.'.format(
            len(concat_dataset), len(loader))
        )
        yield loader


# def get_master_loader(
#         image_lists: List[ImageList],
#         is_train: bool,
#         args: argparse.Namespace,
# ) -> data.DataLoader:
#     print("Creating ConcatDataset")
#     dataset: data.Dataset = data.ConcatDataset(image_lists)
#     print("ConcatDataset finished.")
#     loader = data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         sampler=RandomVidSequenceSampler(dataset, args.frame_len),
#         num_workers=4,
#         drop_last=is_train,
#     ) if is_train else data.DataLoader(
#         dataset,
#         batch_size=args.eval_batch_size,
#         shuffle=False,
#         num_workers=2,
#         drop_last=False,
#     )
#     print(f'Loader for {len(dataset)} images ({len(loader)} batches) created.')
#     return loader


def square_cv2(img: np.ndarray) -> np.ndarray:
    width, height, _ = img.shape
    if width % 16 != 0 or height % 16 != 0:
        img = img[:(width//16)*16, :(height//16)*16]
    return img


def crop_cv2(img: np.ndarray, patch: int) -> np.ndarray:
    height, width, _ = img.shape
    start_x = random.randint(0, height - patch)  # type: ignore
    start_y = random.randint(0, width - patch)  # type: ignore
    return img[  # type: ignore
        start_x: start_x + patch, start_y: start_y + patch
    ]


def multi_crop_cv2(imgs: Tuple[np.ndarray, ...], patch: int) -> Tuple[np.ndarray, ...]:
    height, width, _ = imgs[0].shape
    start_x = random.randint(0, height - patch)  # type: ignore
    start_y = random.randint(0, width - patch)  # type: ignore
    return tuple(
        img[start_x: start_x + patch, start_y: start_y + patch]  # type: ignore
        for img in imgs
    )


def flip_cv2(imgs: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    if random.random() < 0.5:  # type: ignore
        imgs = tuple(np.flip(img, 1) for img in imgs)
    return imgs


def brightness_cv2(imgs: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    brightness_factor = np.random.random() * 0.5 + 0.75
    return tuple((img.astype(np.float32)*brightness_factor).clip(min=0, max=255).astype(img.dtype)
            for img in imgs)


def contrast_cv2(imgs: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    contrast_factor = np.random.random() * 0.5 + 0.75
    out_imgs = []
    for img in imgs:
        im = img.astype(np.float32)
        mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
        im = (1-contrast_factor)*mean + contrast_factor * im
        im = im.clip(min=0, max=255)
        out_imgs.append(im.astype(img.dtype))
    return tuple(out_imgs)


def np_to_torch(img: np.ndarray) -> torch.Tensor:
    img = np.swapaxes(img, 0, 1)  # w, h, 9
    img = np.swapaxes(img, 0, 2)  # 9, h, w
    return torch.from_numpy(img).float()


# class MultiVidDataset(data.Dataset):
#     def __init__(self, id_to_image_lists: Dict[str, data.Dataset]) -> None:
#         super().__init__()
#         self.id_to_image_lists = id_to_image_lists

#     def __getitem__(self, index: int) -> List[torch.Tensor]:
#         # num_of_datasets x frame_len
#         dataset_by_frames = [image_list[index]
#                              for _, image_list in self.id_to_image_lists.items()]
#         # frame_len x num_of_datasets
#         frames_list = zip(*dataset_by_frames)
#         return [torch.cat(frames) for frames in frames_list]
#
#     def __len__(self) -> int:
#         return 0


def load_hkl_images(filepath: str) -> List[Tuple[np.ndarray, ...]]:
    print(f"Loading {filepath}.")
    image_list = hkl.load(filepath)
    return image_list


def convert_images_to_datasets(
        image_list: List[Tuple[np.ndarray, ...]],
        is_train: bool,
        args: argparse.Namespace,
        frame_len: int,
        sampling_range: int,
) -> List[ImageList]:
    datasets = [ImageList(
        imgs, is_train, args, frame_len, sampling_range,
    ) for imgs in image_list]
    return datasets
