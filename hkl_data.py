import os
from typing import Dict, Tuple

import hickle as hkl
import numpy as np
from pympler import asizeof

from waveone.dataset import get_id_to_image_lists, default_loader

PATH = "/scratch/cluster/cywu/kinetics_train_8_100frames_352x288"
STORE = '/scratch/cluster/scottcao/kinetics_8_hkl'
NAME = "train"
MAX_SIZE = 8 * (2 ** 30)


def main() -> None:
    counter = 0
    id_to_image_lists = get_id_to_image_lists(PATH)
    id_to_images: Dict[str, Tuple[np.ndarray, ...]] = {}
    for vid_id, image_list in id_to_image_lists.items():

        hkl_size = asizeof.asizeof(id_to_images)
        if hkl_size >= 8 * (2 ** 30):
            hkl_path = os.path.join(STORE, f"{NAME}{counter}.hkl")
            hkl.dump(id_to_images, hkl_path, mode='w')
            id_to_images = {}
            counter += 1
            print(f"Dumped {hkl_path}. Size {hkl_size}.")

        new_id = "_".join(vid_id)
        images = tuple(default_loader(image_path) for image_path in image_list)
        id_to_images[new_id] = images

        print(f"Finished reading {new_id} into memory.")


if __name__ == '__main__':
    main()
