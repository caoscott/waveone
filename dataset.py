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


def brightness_cv2(imgs):
    brightness_factor = np.random.random() * 0.5 + 0.75
    return [(img*brightness_factor).clip(min=0, max=255).astype(img.dtype)
            for img in imgs]


def contrast_cv2(imgs):
    contrast_factor = np.random.random() * 0.5 + 0.75
    out_imgs = []
    for img in imgs:
        mean = round(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())
        im = (1-contrast_factor)*mean + contrast_factor * im
        im = im.clip(min=0, max=255)
        out_imgs.append(im.astype(img.dtype))
    return out_imgs


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
                img = default_loader(filename).astype(np.float64)
                assert img.max() <= 0.5
                assert img.min() >= -0.5
                self.imgs.append(img)
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
        # if self.frame_len > 1:
        #     index1 = index // self.frame_len
        #     index2 = index % self.frame_len
        #     return [self.imgs[index1], self.imgs[index2]]
        # else:
        #     imgs = [self.imgs[index], self.imgs[index + 1]]

        if self.is_train:
            # If use_bmv, * -1.0 on bmv for flipped images.
            imgs = contrast_cv2(brightness_cv2(flip_cv2(imgs)))

        # CV2 cropping in CPU is faster.
        if self.args.patch and self.is_train:
            # imgs = multi_crop_cv2(imgs, self.args.patch + 1)
            # imgs = [crop_cv2(img, self.args.patch) for img in imgs]
            imgs = multi_crop_cv2(imgs, self.args.patch)

        imgs = tuple(np_to_torch(img / 255 - 0.5) for img in imgs)

        assert len(imgs) == self.frame_len
        return imgs

    def __len__(self):
        # return len(self.imgs) * (len(self.imgs) - 1) // 2 \
        #     if self.frame_len > 1 else len(self.imgs) - 1
        length = self.sampling_range or self.frame_len
        return (0 if len(self.imgs) < length
                else len(self.imgs) - length + 1)
