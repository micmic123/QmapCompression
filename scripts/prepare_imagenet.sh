#!/bin/bash
imagenet_val_dir=$1

python get_paths_imagenet_subset.py --src="$imagenet_val_dir" --list="../data/imagenet_subset_list.csv"
