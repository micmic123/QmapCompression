# Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform
![Figure 2](./assets/compressed_images_with_various_qmaps.svg)
This repository is the implementation of ["Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform"](https://arxiv.org/abs/2108.09551) (ICCV 2021).
Our code is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI).

> **Abstract:** 
>We propose a versatile deep image compression network based on Spatial Feature Transform ([SFT](https://arxiv.org/abs/1804.02815)), which takes a source image and a corresponding quality map as inputs and produce a compressed image with variable rates. 
>Our model covers a wide range of compression rates using a single model, which is controlled by arbitrary pixel-wise quality maps. 
>In addition, the proposed framework allows us to perform task-aware image compressions for various tasks, e.g., classification, by efficiently estimating optimized quality maps specific to target tasks for our encoding network. 
>This is even possible with a pretrained network without learning separate models for individual tasks. 
>Our algorithm achieves outstanding rate-distortion trade-off compared to the approaches based on multiple models that are optimized separately for several different target rates. 
>At the same level of compression, the proposed approach successfully improves performance on image classification and text region quality preservation via task-aware quality map estimation without additional model training. 


## Installation
We tested our code in ubuntu 16.04, g++ 8.4.0, cuda 10.1, python 3.8.8, pytorch 1.7.1.
A C++ 17 compiler is required to use the Range Asymmetric Numeral System implementation.

1. Check your g++ version >= 7. If not, please update it first and make sure to use the updated version.
    - `$ g++ --version`

2. Set up the python environment (Python 3.8).
    
3. Install needed packages.
    - `$ pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
    - `$ pip install -r requirements.txt`
    - If some errors occur in installing [CompressAI](https://github.com/InterDigitalInc/CompressAI), please install it yourself. 
    It is for the entropy coder.
        - We used [CompressAI 1.0.9](https://github.com/micmic123/CompressAI) version.

## Dataset
- Training set: [COCO dataset](https://cocodataset.org/#download)
- Test set: [Kodak dataset](http://r0k.us/graphics/kodak/)

1. (Training set) Download the following files and decompress them.
    - 2014 Train images [83K/13GB]
    - 2014 Train/Val annotations [241MB]
        - instances_train2014.json
    - 2017 Train images [118K/18GB]
    - 2017 Train/Val annotations [241MB]
        - instances_train2017.json

2. (Test set) Download Kodak dataset.
3. Make a directory of structure as follows for the datasets.
```
├── your_dataset_root
    ├── coco
        |── annotations
            ├── instances_train2014.json
            └── instances_train2017.json
        ├── train2014
        └── train2017
    └── kodak
            ├── 1.png
            ├── ...
```
4. Run following command in `scripts` directory.
    - `$ ./prepare.sh your_dataset_root/coco your_dataset_root/kodak`
    - `trainset_coco.csv` and `kodak.csv` will be created in `data` directory.

## Training
### Configuration
We used the same configuration as `./configs/config.yaml` to train our model.
You can change it as you want.
We expect that larger number of training iteration will lead to the better performance.

### Train
`$ python train.py --config=./configs/config.yaml --name=your_instance_name` \
The checkpoints of the model will be saved in `./results/your_instance_name/snapshots`. \
Training for 2M iterations will take about 2-3 weeks on a single GPU like Titan Xp.
At least 12GB GPU memory is needed for the default training setting.

### Resume from a checkpoint
`$ python train.py --resume=./results/your_instance_name/snapshots/your_snapshot_name.pt` \
By default, the original configuration of the checkpoint `./results/your_instance_name/config.yaml` will be used.

## Evaluation
`$ python eval.py --snapshot=./results/your_instance_name/snapshots/your_snapshot_name.pt --testset=./data/kodak.csv`

### Pretrained model
We release the [pretrained model](https://drive.google.com/file/d/1TgCHlA4J2r_566XyfELl-BbANygVf9_u/view?usp=sharing).
Unzip the file and put it in `results` directory.
You can use it like following: \
`$ python eval.py --snapshot=./results/pretrained_dist/snapshots/2M_itrs.pt --testset=./data/kodak.csv`


### Final evaluation results
```
[ Test-1 ] Total: 0.5104 | Real BPP: 0.2362 | BPP: 0.2348 | PSNR: 29.5285 | MS-SSIM: 0.9360 | Aux: 93 | Enc Time: 0.2403s | Dec Time: 0.0356s
[ Test 0 ] Total: 0.2326 | Real BPP: 0.0912 | BPP: 0.0902 | PSNR: 27.1140 | MS-SSIM: 0.8976 | Aux: 93 | Enc Time: 0.2399s | Dec Time: 0.0345s
[ Test 1 ] Total: 0.2971 | Real BPP: 0.1187 | BPP: 0.1176 | PSNR: 27.9824 | MS-SSIM: 0.9159 | Aux: 93 | Enc Time: 0.2460s | Dec Time: 0.0347s
[ Test 2 ] Total: 0.3779 | Real BPP: 0.1559 | BPP: 0.1547 | PSNR: 28.8982 | MS-SSIM: 0.9323 | Aux: 93 | Enc Time: 0.2564s | Dec Time: 0.0370s
[ Test 3 ] Total: 0.4763 | Real BPP: 0.2058 | BPP: 0.2045 | PSNR: 29.9052 | MS-SSIM: 0.9464 | Aux: 93 | Enc Time: 0.2553s | Dec Time: 0.0359s
[ Test 4 ] Total: 0.5956 | Real BPP: 0.2712 | BPP: 0.2697 | PSNR: 30.9739 | MS-SSIM: 0.9582 | Aux: 93 | Enc Time: 0.2548s | Dec Time: 0.0354s
[ Test 5 ] Total: 0.7380 | Real BPP: 0.3558 | BPP: 0.3541 | PSNR: 32.1140 | MS-SSIM: 0.9678 | Aux: 93 | Enc Time: 0.2598s | Dec Time: 0.0358s
[ Test 6 ] Total: 0.9059 | Real BPP: 0.4567 | BPP: 0.4548 | PSNR: 33.2801 | MS-SSIM: 0.9752 | Aux: 93 | Enc Time: 0.2596s | Dec Time: 0.0361s
[ Test 7 ] Total: 1.1050 | Real BPP: 0.5802 | BPP: 0.5780 | PSNR: 34.4822 | MS-SSIM: 0.9811 | Aux: 93 | Enc Time: 0.2590s | Dec Time: 0.0364s
[ Test 8 ] Total: 1.3457 | Real BPP: 0.7121 | BPP: 0.7095 | PSNR: 35.5609 | MS-SSIM: 0.9852 | Aux: 93 | Enc Time: 0.2569s | Dec Time: 0.0367s
[ Test 9 ] Total: 1.6392 | Real BPP: 0.8620 | BPP: 0.8590 | PSNR: 36.5931 | MS-SSIM: 0.9884 | Aux: 93 | Enc Time: 0.2553s | Dec Time: 0.0371s
[ Test10 ] Total: 2.0116 | Real BPP: 1.0179 | BPP: 1.0145 | PSNR: 37.4660 | MS-SSIM: 0.9907 | Aux: 93 | Enc Time: 0.2644s | Dec Time: 0.0376s
[ Test ] Total mean: 0.8841 | Enc Time: 0.2540s | Dec Time: 0.0361s
```

- `[ TestN ]` means to use a uniform quality map of (N/10) value for evaluation. 
    - For example, in the case of `[ Test8 ]`, a uniform quality map of 0.8 is used.
- `[ Test-1 ]` means to use pre-defined non-uniform quality maps for evaluation.
- `Bpp` is the theoretical average bpp calculated by the trained probability model.
- `Real Bpp` is the real average bpp for the saved file including quantized latent representations and metadata.
    - All bpps reported in the paper are `Real Bpp`.
- `Total` is the average loss value.

## Classification-aware compression
### Dataset
We made a test set of ImageNet dataset by sampling 102 categories and choosing 5 images per a category randomly.
1. Prepare the original ImageNet validation set `ILSVRC2012_img_val`.
2. Run following command in `scripts` directory.
    - `$ ./prepare_imagenet.sh your_dataset_root/ILSVRC2012_img_val`
    - `imagenet_subset.csv` will be created in `data` directory.

### Running
`$ python classification_aware.py --snapshot=./results/your_instance_name/snapshots/your_snapshot_name.pt` \
A result plot `./classificatoin_result.png` will be generated.

## Citation
```bibtex
@inproceedings{song2021variable,
  title={Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform},
  author={Song, Myungseo and Choi, Jinyoung and Han, Bohyung},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2380--2389},
  year={2021}
}
```
