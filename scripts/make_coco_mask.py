from pycocotools.coco import COCO
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='path for coco dataset', required=True)
args = parser.parse_args()

data_dir = args.dir  # '/data/micmic123/coco'
data_types = ['train2014', 'train2017']  # val2017
for data_type in data_types:
    annFile = f'{data_dir}/annotations/instances_{data_type}.json'
    mask_dir = f'{data_dir}/{data_type}_mask'
    os.makedirs(mask_dir, exist_ok=True)

    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)

    cnt = 0
    for info in tqdm(images, desc=mask_dir):
        img_id = info['id']
        img_filename = info['file_name']
        img = Image.open(f'{data_dir}/{data_type}/{img_filename}')
        w, h = img.size

        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)

        mask = np.zeros((), dtype=np.uint8)
        for i in range(len(anns)):
            pixel_value = anns[i]['category_id']
            mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

        if len(np.unique(mask)) < 2 or w < 256 or h < 256:
            cnt += 1
            continue
        filename = img_filename.split('.')[0]
        mask = Image.fromarray(mask)
        mask.save(os.path.join(mask_dir, f'{filename}.png'))

    print('skipped:', cnt, f'{cnt / len(images):.2f}%')
