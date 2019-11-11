import glob
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data


def get_loader(is_train: bool, root: str, frame_len: int, sampling_range: int,
               args) -> data.DataLoader:
    # print('\nCreating loader for %s...' % root)

    dset = ImageFolder(
        is_train=is_train,
        root=root,
        args=args,
        sampling_range=sampling_range,
        frame_len=frame_len,
    )

    loader = data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
        num_workers=0,
        drop_last=is_train,
    )

    print('Loader for {} images ({} batches) created.'.format(
        len(dset), len(loader))
    )

    return loader


def default_loader(path: str):
    cv2_img = cv2.imread(path)
    if cv2_img.shape is None:
        print(path)
        print(cv2_img)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    width, height, _ = cv2_img.shape
    if width % 16 != 0 or height % 16 != 0:
        cv2_img = cv2_img[:(width//16)*16, :(height//16)*16]

    return cv2_img


def crop_cv2(img, patch):
    height, width, _ = img.shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    return img[start_x: start_x + patch, start_y: start_y + patch]


def multi_crop_cv2(imgs, patch):
    height, width, _ = imgs[0].shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)
    return [img[start_x: start_x + patch, start_y: start_y + patch]
            for img in imgs]


def flip_cv2(imgs):
    if random.random() < 0.5:
        imgs = [img[::-1].copy() for img in imgs]

        # assert img.shape[2] == 13, img.shape
        # height first, and then width. but BMV is (width, height)... sorry..
        # img[:, :, 9] = img[:, :, 9] * (-1.0)
        # img[:, :, 11] = img[:, :, 11] * (-1.0)
    return imgs


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, 9
    img = np.swapaxes(img, 0, 2)  # 9, h, w
    return torch.from_numpy(img).float()


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, is_train: bool, root: str, args,
                 frame_len: int = 5, sampling_range: int = 0) -> None:
        self.is_train = is_train
        self.root = root
        self.args = args

        self.patch = args.patch

        assert frame_len > 0
        assert sampling_range == 0 or sampling_range >= frame_len

        self.frame_len = frame_len
        self.sampling_range = sampling_range

        self._load_image_list()

    def _load_image_list(self):
        self.imgs = []
        self.fns = []

        for filename in sorted(glob.iglob(self.root + '/*png')):
            if os.path.isfile(filename):
                self.imgs.append(default_loader(filename).astype(np.float64))
                self.fns.append(filename)

        print('%d images loaded.' % len(self.imgs))

    def __getitem__(self, index):
        imgs = []
        if self.sampling_range:
            offsets = np.random.permutation(
                self.sampling_range)[:self.frame_len]
            imgs = [self.imgs[index + offset] for offset in offsets]
        else:
            imgs = self.imgs[index: index+self.frame_len]

        if self.is_train:
            # If use_bmv, * -1.0 on bmv for flipped images.
            imgs = flip_cv2(imgs)

        # CV2 cropping in CPU is faster.
        if self.patch and self.is_train:
            # imgs = multi_crop_cv2(imgs, self.patch + 1)
            # imgs = [crop_cv2(img, self.patch) for img in imgs]
            imgs = multi_crop_cv2(imgs, self.patch)

        imgs = tuple(np_to_torch(img / 255.0 - 0.5) for img in imgs)
        assert len(imgs) == self.frame_len
        return imgs

    def __len__(self):
        length = self.sampling_range or self.frame_len
        return (0 if len(self.imgs) < length
                else len(self.imgs) - length + 1)
