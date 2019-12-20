import os
from collections import defaultdict
import glob
from typing import DefaultDict, Dict, List, Tuple

import cv2
import hickle as hkl
import numpy as np
from pympler import asizeof

# MATCH = "/scratch/cluster/scottcao/vcii_data/train/*.png"
# STORE = '/scratch/cluster/scottcao/vcii_hkl'
# NAME = "train"

MATCH = "/scratch/cluster/scottcao/vcii_data/eval/*.png"
STORE = '/scratch/cluster/scottcao/vcii_hkl'
NAME = "eval"

# MATCH = "/scratch/cluster/cywu/kinetics_train_8_100frames_352x288/6yuov4mSl1Q_*.png"
# STORE = '/scratch/cluster/scottcao/kinetics_8_hkl_subset/'
# NAME = "kinetics"

# MATCH = "/scratch/cluster/cywu/kinetics_train_8_100frames_352x288/*.png"
# STORE = '/scratch/cluster/scottcao/kinetics_8_hkl'
# NAME = "kinetics"
MAX_SIZE = 13 * (2 ** 30)


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
        id_to_images: List[Tuple[np.ndarray, ...]],
        counter: int,
        obj_size: int,
) -> str:
    hkl_path = os.path.join(STORE, f"{NAME}{counter}.hkl")
    hkl.dump(id_to_images, hkl_path, mode="w", compression="gzip")
    print(f"Dumped {hkl_path}. Size in memory: {obj_size / 2 ** 30 :.6f} GB. "
          f"Size on disk: {os.path.getsize(hkl_path) / 2 ** 30 :.6f} GB. ")
    return hkl_path


def check_load(path: str) -> None:
    hkl.load(path)


def main() -> None:
    counter = 0
    id_to_image_lists = get_id_to_image_lists(MATCH)
    image_lists: List[Tuple[np.ndarray, ...]] = []

    for vid_id, image_list in id_to_image_lists.items():
        obj_size = asizeof.asizeof(image_lists)
        if obj_size >= 8 * (2 ** 30):
            file_path = save(image_lists, counter, obj_size)
            image_lists = []
            counter += 1
            check_load(file_path)

        new_id = "_".join(vid_id)
        images = tuple(default_loader(image_path) for image_path in image_list)
        image_lists.append(images)

        print(f"Finished reading {new_id} into memory.")

    if image_lists:
        file_path = save(image_lists, counter, asizeof.asizeof(image_lists))
        check_load(file_path)


if __name__ == '__main__':
    main()
