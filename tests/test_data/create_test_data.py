# This script creates the folders/data needed for some of the integration tests.
# A prerequirement is for the full dataset to exist in the original_dir location, so please
# download the original data manually
# https://www.kaggle.com/competitions/dogs-vs-cats/data

# This snippet has been taken from
# Deep Learning with Python, Francois Chollet, Manning Publications.

import os
import pathlib
import shutil

original_dir = "./dogs-vs-cats/train"
new_base_dir = pathlib.Path("cats_vs_dogs_small")


def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir, exist_ok=True)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=os.path.join(original_dir, fname),
                            dst=os.path.join(dir, fname))


make_subset("train", start_index=0, end_index=100)
make_subset("validation", start_index=1000, end_index=1100)
make_subset("test", start_index=1500, end_index=1600)
