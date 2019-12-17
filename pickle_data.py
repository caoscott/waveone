import os
from collections import defaultdict
import glob
from typing import DefaultDict, Dict, List, Tuple

import cv2
import torch
import numpy as np
from pympler import asizeof

# MATCH = "/scratch/cluster/scottcao/vcii_all/*.png"
# STORE = '/scratch/cluster/scottcao/vcii_pkl'
# NAME = "vcii"

# MATCH = "/scratch/cluster/cywu/kinetics_train_8_100frames_352x288/6yuov4mSl1Q_*.png"
# STORE = '/scratch/cluster/scottcao/kinetics_8_pkl_subset/'
# NAME = "kinetics"

MATCH = "/scratch/cluster/cywu/kinetics_train_8_100frames_352x288/*.png"
STORE = '/scratch/cluster/scottcao/kinetics_8_pkl'
NAME = "kinetics"
MAX_SIZE = 8 * (2 ** 30)


def default_loader(path: str) -> np.ndarray:
    cv2_img = cv2.imread(path)
    if cv2_img.shape is None:
        print(path)
        print(cv2_img)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return cv2_img


def get_id_to_image_lists(root: str) -> Dict[Tuple[str, ...], List[str]]:
    print(f'Loading {root}')
    id_to_images: DefaultDict[Tuple[str, ...], List[str]] = defaultdict(list)
    for filename in glob.iglob(root):
        vid_id = tuple(filename.split("_")[:-1])
        id_to_images[vid_id].append(filename)
    for vid_id in id_to_images:
        id_to_images[vid_id].sort()
    print(f"Finished loading {root}.")
    return id_to_images


def save(
        id_to_images: Dict[str, Tuple[np.ndarray, ...]],
        counter: int,
        obj_size: int,
) -> None:
    pkl_path = os.path.join(STORE, f"{NAME}{counter}.pkl")
    torch.save(id_to_images, pkl_path)
    print(f"Dumped {pkl_path}. Size {obj_size}.")


def main() -> None:
    counter = 0
    id_to_image_lists = get_id_to_image_lists(MATCH)
    id_to_images: Dict[str, Tuple[np.ndarray, ...]] = {}

    for vid_id, image_list in id_to_image_lists.items():
        obj_size = asizeof.asizeof(id_to_images)
        if obj_size >= 8 * (2 ** 30):
            save(id_to_images, counter, obj_size)
            id_to_images = {}
            counter += 1

        new_id = "_".join(vid_id)
        images = tuple(default_loader(image_path) for image_path in image_list)
        id_to_images[new_id] = images

        print(f"Finished reading {new_id} into memory.")

    if id_to_images:
        save(id_to_images, counter, asizeof.asizeof(id_to_images))


if __name__ == '__main__':
    main()
