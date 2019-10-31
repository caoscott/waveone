import glob
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data


def get_loader(is_train: bool, root: str, mv_dir: str, args) -> data.DataLoader:
    # print('\nCreating loader for %s...' % root)

    dset = ImageFolder(
        is_train=is_train,
        root=root,
        mv_dir=mv_dir,
        args=args,
    )

    loader = data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=False,
        num_workers=1,
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


def read_bmv(fn):
    a = cv2.imread(fn, 0)
    if a is not None:
        width, height = a.shape
        if width % 16 != 0 or height % 16 != 0:
            a = a[:(width//16)*16, :(height//16)*16]

        return a[:, :, np.newaxis].astype(float) - 128.0
    else:
        print('no bmv found (it\'s okay if not too often)', fn)
        return None


def get_bmv(img, fns):
    before_x, before_y, after_x, after_y = fns

    bmvs = [read_bmv(before_x),
            read_bmv(before_y),
            read_bmv(after_x),
            read_bmv(after_y)]

    if bmvs[0] is None or bmvs[1] is None:
        if 'ultra_video_group' in before_x:
            # We need HW to be (16n1, 16n2).
            bmvs[0] = np.zeros((1072, 1920, 1))
            bmvs[1] = np.zeros((1072, 1920, 1))
        else:
            bmvs[0] = np.zeros((288, 352, 1))
            bmvs[1] = np.zeros((288, 352, 1))
    else:
        bmvs[0] = bmvs[0] * (-2.0)
        bmvs[1] = bmvs[1] * (-2.0)

    if bmvs[2] is None or bmvs[3] is None:
        if 'ultra_video_group' in before_x:
            bmvs[2] = np.zeros((1072, 1920, 1))
            bmvs[3] = np.zeros((1072, 1920, 1))
        else:
            bmvs[2] = np.zeros((288, 352, 1))
            bmvs[3] = np.zeros((288, 352, 1))
    else:
        bmvs[2] = bmvs[2] * (-2.0)
        bmvs[3] = bmvs[3] * (-2.0)

    return bmvs


def crop_cv2(img, patch):
    height, width, c = img.shape
    start_x = random.randint(0, height - patch)
    start_y = random.randint(0, width - patch)

    return img[start_x: start_x + patch, start_y: start_y + patch]


def flip_cv2(img):
    if random.random() < 0.5:
        img = img[::-1].copy()

        # assert img.shape[2] == 13, img.shape
        # height first, and then width. but BMV is (width, height)... sorry..
        # img[:, :, 9] = img[:, :, 9] * (-1.0)
        # img[:, :, 11] = img[:, :, 11] * (-1.0)
    return img


# (Close, far)
def get_group_filenames(filename, img_idx, distance1, distance2):
    dtype = filename[-3:]
    assert filename[-4] == '.'
    code = filename[:-4].split('_')[-1]

    # I 2 3 D 5 6 D 8 9 B 11 12 I
    if img_idx % 12 in [3, 6, 9, 0]:
        delta_close = distance1
        delta_far = distance2 * (-1)
    else:
        delta_close = distance1 * (-1)
        delta_far = distance2

    filenames = [filename[:-4 - len(code)] + str(img_idx + delta_close).zfill(len(code)) + '.%s' % dtype,
                 filename[:-4 - len(code)] +
                 str(img_idx).zfill(len(code)) + '.%s' % dtype,
                 filename[:-4 - len(code)] + str(img_idx + delta_far).zfill(len(code)) + '.%s' % dtype]

    return filenames


def get_bmv_filenames(mv_dir, main_fn):

    fn = main_fn.split('/')[-1][:-4]

    return (os.path.join(mv_dir, fn + '_before_flow_x_0001.jpg'),
            os.path.join(mv_dir, fn + '_before_flow_y_0001.jpg'),
            os.path.join(mv_dir, fn + '_after_flow_x_0001.jpg'),
            os.path.join(mv_dir, fn + '_after_flow_y_0001.jpg'))


def get_identity_grid(shape):
    width, height = shape
    grid = np.zeros((width, height, 2))
    for i in range(width):
        for j in range(height):
            grid[i, j, 0] = float(j) * (2.0 / (height - 1.0)) - 1.0
            grid[i, j, 1] = float(i) * (2.0 / (width - 1.0)) - 1.0
    return grid


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, 9
    img = np.swapaxes(img, 0, 2)  # 9, h, w
    return torch.from_numpy(img).float()


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, is_train: bool, root: str, mv_dir: str, args):

        self.is_train = is_train
        self.root = root
        self.args = args
        self.mv_dir = mv_dir

        self.patch = args.patch
        self.loader = default_loader
        self.v_compress = args.v_compress
        self._num_crops = args.num_crops

        self.identity_grid = None

        self._load_image_list()

        # print('\tdistance=%d/%d' % (args.distance1, args.distance2))

    def _load_image_list(self):
        self.imgs = []
        # dist1, dist2 = self.args.distance1, self.args.distance2

        # if self.v_compress:
        #     if dist1 == 6 and dist2 == 6:
        #         positions = [7]
        #     elif dist1 == 3 and dist2 == 3:
        #         positions = [4, 10]
        #     elif dist1 == 1 and dist2 == 2:
        #         positions = [2, 3, 5, 6, 8, 9, 11, 0]
        #     else:
        #         assert False, 'not implemented.'

        for filename in sorted(glob.iglob(self.root + '/*png')):
            # img_idx = int(filename[:-4].split('_')[-1])

            # if self.args.v_compress:
            #     if all(os.path.isfile(fn) for fn in
            #            get_group_filenames(
            #             filename, img_idx, dist1, dist2)):
            #         self.imgs.append(filename)
            # else:
            #     if (img_idx % 12) != 1:
            #         continue
            if os.path.isfile(filename):
                self.imgs.append(
                    (self.loader(filename).astype(np.float64), filename))

        print('%d images loaded.' % len(self.imgs))

    def get_group_data(self, filename):
        img_idx = int(filename[:-4].split('_')[-1])

        filenames = get_group_filenames(
            filename, img_idx,
            self.args.distance1,
            self.args.distance2)
        assert all(os.path.isfile(fn) for fn in filenames), filenames
        assert len(filenames) == 3

        imgs_ = [self.loader(fn).astype(np.float64) for fn in filenames]

        main_fn = filenames[1]
        return np.concatenate(imgs_, axis=2), main_fn

    def get_frame_data(self, filename):
        img = self.loader(filename)
        return img, filename

    def __getitem__(self, index):
        img1, fn1 = self.imgs[index]
        img2, fn2 = self.imgs[index + 1]
        img = np.concatenate((img1, img2), axis=2).astype(np.float64)

        assert img.shape[2] == 6
        if self.is_train:
            # If use_bmv, * -1.0 on bmv for flipped images.
            img = flip_cv2(img)

        # CV2 cropping in CPU is faster.
        if self.is_train and self.patch:
            img[:3] = crop_cv2(img[:3], self.patch)
            img[3:] = crop_cv2(img[3:], self.patch)

        img /= 255.0
        img = np_to_torch(img)

        frame1, frame2 = img[:3], img[3:]

        return frame1, frame2, fn1, fn2

    def __len__(self):
        return 0 if len(self.imgs) == 0 else len(self.imgs) - 1
