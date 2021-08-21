#!/bin/bash
coco_dir=$1
kodak_dir=$2

python make_coco_mask.py --dir="$coco_dir"

python get_paths_coco.py --dir="$coco_dir"

python get_paths.py --dir="$kodak_dir" --name=kodak
