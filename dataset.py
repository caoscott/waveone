import argparse
import copy
import glob
import os
import random
import time
from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, List, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data


class ImageList(data.Dataset):
    def __init__(
        self,
        img_paths: List[str],
        is_train: bool,
        args: argparse.Namespace,
        frame_len: int,
        sampling_range: int,
    ) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.is_train = is_train
        self.frame_len = frame_len
        self.padding_len = max(frame_len - len(img_paths), 0)
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
            indexed_paths = self.img_paths
        elif self.sampling_range > 0:
            idx_sampling_range = min(
                self.sampling_range, len(self.img_paths)-index)
            offsets = np.random.permutation(idx_sampling_range)[
                :self.frame_len]
            indexed_paths = [self.img_paths[index + offset]
                             for offset in np.sort(offsets)]
        else:
            indexed_paths = self.img_paths[index: index+self.frame_len]

        images = tuple(default_loader(image_name)
                       for image_name in indexed_paths)
        if self.is_train:
            # images = contrast_cv2(brightness_cv2(flip_cv2(images)))
            images = flip_cv2(images)
            if self.args.patch:
                images = multi_crop_cv2(images, self.args.patch)
        images = tuple(square_cv2(img) for img in images)

        frames: Tuple[torch.Tensor, ...] = tuple(
            np_to_torch(img.astype(np.float64)/255*2 - 1) for img in images
        )
        existence_mask: Tuple[torch.Tensor, ...] = tuple(
            torch.ones((1, 1, 1, 1))) * len(images)  # type: ignore

        if self.padding_len > 0:
            frames += tuple(
                torch.zeros_like(frames[0])) * self.padding_len  # type: ignore
            existence_mask += (tuple(
                torch.zeros((1, 1, 1, 1))) * self.padding_len  # type: ignore
            )

        if self.args.network == "opt":
            for frame in frames:
                assert frame.max() <= 1  # type: ignore
                assert frame.min() >= -1  # type: ignore
        assert len(frames) == self.frame_len, \
            f"{len(frames)} != {self.frame_len}"
        assert len(frames) == len(existence_mask), \
            f"{len(frames)} != {len(existence_mask)}"

        return frames, existence_mask

    def __len__(self) -> int:
        if self.padding_len > 0:
            return 1
        return len(self.img_paths) - self.frame_len + 1


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


def get_vid_id(filename: str) -> str:
    return "_".join(filename.split("_")[:-1])


def get_loader(
        paths: List[str],
        is_train: bool,
        args: argparse.Namespace,
) -> data.DataLoader:
    id_to_filepaths = get_id_to_filepaths(is_train, paths)
    max_frame_len = max(len(filepaths)
                        for filepaths in id_to_filepaths.values())
    datasets = tuple(
        ImageList(
            filepaths,
            is_train=is_train,
            args=args,
            frame_len=args.frame_len if is_train else max_frame_len,
            sampling_range=args.sampling_range if is_train else 0,
        ) for filepaths in id_to_filepaths.values()
    )
    concat_dataset: data.Dataset = data.ConcatDataset(datasets)
    loader = data.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        sampler=RandomVidSequenceSampler(concat_dataset, args.frame_len),
        num_workers=6,
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
    return loader


def default_loader(path: str) -> np.ndarray:
    cv2_img = cv2.imread(path)
    if cv2_img.shape is None:
        print(path)
        print(cv2_img)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return cv2_img


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


def get_id_to_filepaths(
        is_train: bool, paths: List[str],
) -> Dict[str, List[str]]:
    id_to_file_paths: DefaultDict[str, List[str]] = defaultdict(list)
    for filepath in paths:
        vid_id = get_vid_id(filepath)
        id_to_file_paths[vid_id].append(filepath)
    for file_paths in id_to_file_paths.values():
        file_paths.sort()
    return id_to_file_paths
