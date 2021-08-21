import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src', help='source dir', required=True)
parser.add_argument('--list', help='csv file of file names', required=True)
args = parser.parse_args()

data_dir = '../data'
os.makedirs(data_dir, exist_ok=True)

df = pd.read_csv(args.list)
names = df['name'].tolist()
labels = df['label'].tolist()

dataset_paths = []
labels_new = []
for name, label in zip(names, labels):
    dataset_paths.append(os.path.join(args.src, name))
    labels_new.append(label)

pd.DataFrame({'path': dataset_paths, 'label': labels_new}).to_csv(f'{data_dir}/imagenet_subset.csv', index=False)
print(len(dataset_paths))
