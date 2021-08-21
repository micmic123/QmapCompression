import pandas as pd
import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='path for coco dataset', required=True)
args = parser.parse_args()

data_dir = '../data'
coco_dir = args.dir
os.makedirs(data_dir, exist_ok=True)

mask_filtered_path = []
img_paths = []
for name in ['train2014', 'train2017']:
    mask_paths = glob.glob(os.path.join(coco_dir, f'{name}_mask/*'))
    for mask_path in tqdm(mask_paths, desc=name):
        filename = os.path.basename(mask_path).rsplit('.', 1)[0]
        img = Image.open(mask_path)
        w, h = img.size
        if w < 256 or h < 256:
            continue

        mask_filtered_path.append(mask_path)
        img_paths.append(os.path.join(coco_dir, name, f'{filename}.jpg'))

print(len(img_paths))
pd.DataFrame({'path': img_paths, 'seg_path': mask_filtered_path}).to_csv(f'{data_dir}/trainset_coco.csv', index=False)
