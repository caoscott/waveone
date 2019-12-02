import argparse
import glob
import os
import os.path
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List

import cv2
import numpy as np
import torch
import torch.utils.data as data


def get_vid_id(filename: str) -> str:
    return "_".join(filename.split("_")[:-1])


def get_loaders(
        is_train: bool,
        root: str,
        frame_len: int,
        sampling_range: int,
        args: argparse.Namespace,
) -> Dict[str, data.DataLoader]:
    print('Creating loaders for %s...' % root)
    id_to_image_lists = get_id_to_image_lists(
        is_train, root, args, frame_len, sampling_range)
    id_to_loaders: Dict[str, data.DataLoader] = {}
    for vid_id, image_list in id_to_image_lists.items():
        loader = data.DataLoader(
            image_list,
            batch_size=args.batch_size if is_train else args.eval_batch_size,
            shuffle=is_train,
            num_workers=2,
            drop_last=is_train,
        )
        id_to_loaders[vid_id] = loader
        print('Loader for {} images ({} batches) created.'.format(
            len(image_list), len(loader))
        )

    return id_to_loaders


def get_master_loader(
        is_train: bool,
        root: str,
        frame_len: int,
        sampling_range: int,
        args: argparse.Namespace,
) -> data.DataLoader:
    print(f'Creating loader for {root}')
    id_to_image_lists: Dict[str, ImageList] = get_id_to_image_lists(
        is_train, root, args, frame_len, sampling_range)
    dataset: data.Dataset = data.ConcatDataset(
        [image_list for _, image_list in id_to_image_lists.items()]
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
        num_workers=2,
        drop_last=is_train,
    )
    print('Loader for {} images ({} batches) created.'.format(
        len(dataset), len(loader))
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


def multi_crop_cv2(imgs: List[np.ndarray], patch: int) -> List[np.ndarray]:
    height, width, _ = imgs[0].shape
    start_x = random.randint(0, height - patch)  # type: ignore
    start_y = random.randint(0, width - patch)  # type: ignore
    return [img[start_x: start_x + patch, start_y: start_y + patch]  # type: ignore
            for img in imgs]


def flip_cv2(imgs: List[np.ndarray]) -> List[np.ndarray]:
    if random.random() < 0.5:  # type: ignore
        imgs = [img[::-1].copy() for img in imgs]
    return imgs


def brightness_cv2(imgs: List[np.ndarray]) -> List[np.ndarray]:
    brightness_factor = np.random.random() * 0.5 + 0.75
    return [(img.astype(np.float32)*brightness_factor).clip(min=0, max=255).astype(img.dtype)
            for img in imgs]


def contrast_cv2(imgs: List[np.ndarray]) -> List[np.ndarray]:
    contrast_factor = np.random.random() * 0.5 + 0.75
    out_imgs = []
    for img in imgs:
        im = img.astype(np.float32)
        mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
        im = (1-contrast_factor)*mean + contrast_factor * im
        im = im.clip(min=0, max=255)
        out_imgs.append(im.astype(img.dtype))
    return out_imgs


def np_to_torch(img: np.ndarray) -> torch.Tensor:
    img = np.swapaxes(img, 0, 1)  # w, h, 9
    img = np.swapaxes(img, 0, 2)  # 9, h, w
    return torch.from_numpy(img).float()


class MultiVidDataset(data.Dataset):
    def __init__(self, id_to_image_lists: Dict[str, data.Dataset]) -> None:
        super().__init__()
        self.id_to_image_lists = id_to_image_lists

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        # num_of_datasets x frame_len
        dataset_by_frames = [image_list[index]
                             for _, image_list in self.id_to_image_lists.items()]
        # frame_len x num_of_datasets
        frames_list = zip(*dataset_by_frames)
        return [torch.cat(frames) for frames in frames_list]

    def __len__(self) -> int:
        return 0


class ImageList(data.Dataset):
    def __init__(
        self,
        imgs: List[np.ndarray],
        is_train: bool,
        args: argparse.Namespace,
        frame_len: int,
        sampling_range: int,
    ) -> None:
        super().__init__()
        self.imgs = imgs
        self.is_train = is_train
        self.frame_len = frame_len
        self.sampling_range = sampling_range
        self.args = args

        assert frame_len > 0
        assert sampling_range == 0 or sampling_range >= frame_len
        assert len(self.imgs) >= self.frame_len

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        imgs: List[np.ndarray] = []
        if self.sampling_range:
            idx_sampling_range = min(self.sampling_range, len(self.imgs)-index)
            offsets = np.random.permutation(idx_sampling_range)[:self.frame_len]
            imgs = [self.imgs[index + offset] for offset in offsets]
        else:
            imgs = self.imgs[index: index+self.frame_len]

        if self.is_train:
            # imgs = contrast_cv2(brightness_cv2(flip_cv2(imgs)))
            imgs = flip_cv2(imgs)
            if self.args.patch:
                imgs = multi_crop_cv2(imgs, self.args.patch)
        imgs = [square_cv2(img) for img in imgs]

        frames: List[torch.Tensor] = [np_to_torch(img.astype(np.float64)/255*2 - 1)
                                      for img in imgs]

        for frame in frames:
            assert frame.max() <= 1  # type: ignore
            assert frame.min() >= -1  # type: ignore
        assert len(frames) == self.frame_len

        return frames

    def __len__(self) -> int:
        return len(self.imgs) - self.frame_len + 1


def get_id_to_image_lists(
        is_train: bool, root: str, args: argparse.Namespace,
        frame_len: int, sampling_range: int
) -> Dict[str, ImageList]:
    id_to_images: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    for filename in sorted(glob.iglob(root + '/*png')):
        if os.path.isfile(filename):
            vid_id = "_".join(filename.split("_")[:-1])
            img = default_loader(filename)
            id_to_images[vid_id].append(img)
    id_to_datasets: Dict[str, ImageList] = {}
    for vid_id, imgs in id_to_images.items():
        id_to_datasets[vid_id] = ImageList(
            imgs, is_train, args, frame_len, sampling_range
        )
    return id_to_datasets
