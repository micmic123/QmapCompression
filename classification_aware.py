import argparse
import sys
import os
from time import time

import torch
from torchvision import transforms
import torchvision.models as models

from gradcam import GradCAM, GradCAMpp

from models.models import SpatiallyAdaptiveCompression
from dataset import ImagenetDataset
from utils import load_checkpoint, AverageMeter, get_config, _encode, _decode


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Classification-aware compression')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='file name for result plot', default='classification_result', type=str)
    parser.add_argument('--imagenet', help='imagenet subset path', type=str, default='./data/imagenet_subset.csv')
    parser.add_argument('--cuda', help='use cuda', action='store_true', default=True)
    parser.add_argument('--snapshot', help='snapshot path', type=str)
    args = parser.parse_args(argv)

    assert args.snapshot.startswith('./')
    dir_path = '/'.join(args.snapshot.split('/')[:-2])
    args.config = os.path.join(dir_path, 'config.yaml')

    return args


def init(x, label, lmbda=0.1):
    qmap = torch.empty_like(x)[:, 0:1, :, :].normal_(mean=-2, std=1)
    qmap = qmap.clone().detach().to(device)
    qmap.requires_grad_()
    optimizer = torch.optim.LBFGS([qmap], max_iter=1)
    train_param = {
        'x': x,
        'label': label,
        'i': 0,
        'loss_best': float('inf'),
        'score_best': 0,
        'ce_best': float('inf'),
        'qmap_mean_best': 0,
        'qmap': qmap,
        'qmap_best': qmap,
        'i_best': 0,
        'optimizer': optimizer,
        'bpp_best': float('inf'),
        'topk_indices_best': [],
        'topk_score_best': [],
        'lmbda': lmbda
    }
    return train_param


def closure_():
    global train_param, w
    x = train_param['x']
    label = train_param['label']
    qmap = train_param['qmap']

    qmap_norm = normalize_qmap(qmap)
    out_net = model(x, qmap_norm)
    x_recon = torch.clamp(out_net['x_hat'], 0, 1)

    bpp = compute_loss_bpp_(out_net)
    pred = vgg16(normalize(x_recon))
    ce = criterion(pred, label)
    score = pred.flatten()[label]

    loss = w * (train_param['lmbda'] * ce + bpp)

    train_param['optimizer'].zero_grad()
    loss.backward()
    train_param['i'] += 1

    qmap_mean = torch.mean(qmap_norm)
    topk = torch.topk(vgg16(normalize(x_recon)), 5)
    topk_indices = topk[1].cpu().tolist()
    topk_score = topk[0].cpu().tolist()

    if train_param['loss_best'] > loss or train_param['i'] == 1:
        train_param['loss_best'] = loss
        train_param['ce_best'] = ce
        train_param['score_best'] = score
        train_param['qmap_best'] = qmap.clone().detach()
        train_param['qmap_mean_best'] = qmap_mean
        train_param['i_best'] = train_param['i']
        train_param['bpp_best'] = bpp.clone().cpu().detach()
        train_param['topk_indices_best'] = topk_indices
        train_param['topk_score_best'] = topk_score

    torch.nn.utils.clip_grad_norm_(qmap, grad_clip)

    return loss


def optimize(train_param, total_itr=200):
    while train_param['i'] < total_itr:
        train_param['optimizer'].step(closure_)


def compute_loss_bpp_(out_net):
    N, _, H, W = out_net['x_hat'].size()
    num_pixels = N * H * W
    return sum((-torch.log2(likelihoods).sum() / num_pixels)
               for likelihoods in out_net['likelihoods'].values())


def recon_uniform(model, img, q=0.1):
    qmap = q * torch.ones_like(img)[:, 0:1, :, :]
    qmap = qmap.to(device)
    bpp, out, enc_time = _encode(model, img, tmp_path, qmap)
    x_hat, dec_time = _decode(model, tmp_path, coder='ans', verbose=False)

    return x_hat, bpp


def recon_with_qmap(model, img, qmap):
    bpp, out, enc_time = _encode(model, img, tmp_path, qmap)
    x_hat, dec_time = _decode(model, tmp_path, coder='ans', verbose=False)

    return x_hat, bpp


def eval_classification(classifier, img, label):
    pred = classifier(normalize(img))
    topk = torch.topk(pred, 5)
    topk_indices = topk[1].cpu().tolist()[0]

    top1 = (topk_indices[0] == label.item())
    top5 = (label.item() in topk_indices[:5])

    return top1, top5


def get_grad_cam(x):
    mask, _ = camera(normalize(x))
    return mask


def normalize_qmap(qmap):
    return torch.sigmoid(qmap)


def plot(result):
    import matplotlib.pyplot as plt

    plt.style.use('default')
    plt.style.use('seaborn-white')
    plt.rcParams['axes.titlesize'] = 45
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.labelsize'] = 48
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelpad'] = 14
    plt.rcParams['axes.edgecolor'] = 'lightgrey'
    plt.rcParams['grid.color'] = 'whitesmoke'
    plt.rcParams['xtick.labelsize'] = 32
    plt.rcParams['xtick.major.pad'] = 20
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.labelsize'] = 32
    plt.rcParams['ytick.major.pad'] = 15
    plt.rcParams['figure.subplot.wspace'] = 0.32
    plt.rcParams['figure.subplot.hspace'] = 0.30
    plt.rcParams['legend.loc'] = 'lower right'
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.edgecolor'] = 'gainsboro'
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.marker'] = 'd'
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['figure.figsize'] = (10, 9)

    plt.plot(result[5]['bpp'], result[5]['acc1'], '-o', label='Optimized in 0.65s@1', color='#ff0000')

    plt.plot(result[5]['bpp'], result[5]['acc5'], '--o', label='Optimized in 0.65s@5', color='#ff0000')

    plt.plot(result[3]['bpp'], result[3]['acc1'], '-^', label='Optimized in 0.37s@1', color='#ffc010')

    plt.plot(result[3]['bpp'], result[3]['acc5'], '--^', label='Optimized in 0.37s@5', color='#ffc010')

    plt.plot(result['cam']['bpp'], result['cam']['acc1'], '-s', label='Grad-CAM@1', color='#00ff00')

    plt.plot(result['cam']['bpp'], result['cam']['acc5'], '--s', label='Grad-CAM@5', color='#00ff00')

    plt.plot(result['uniform']['bpp'], result['uniform']['acc1'], '', label='Uniform@1', color='#0000ff')

    plt.plot(result['uniform']['bpp'], result['uniform']['acc5'], '--', label='Uniform@5', color='#0000ff')

    plt.plot([0, 1.14], [x for x in result['original']['acc1'] for i in range(2)], '-', color='#555555')

    plt.plot([0, 1.14], [x for x in result['original']['acc5'] for i in range(2)], '--', color='#555555')

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid()
    plt.legend()
    plt.xlabel('Bits per pixel (BPP)', fontsize=32)
    plt.ylabel('Accuaracy', fontsize=32)
    plt.savefig(f'./{args.name}.png', bbox_inches='tight')


def result_init(result, name):
    result[name] = {'bpp': [], 'acc1': [], 'acc5': []}


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    config = get_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmp_path = '/tmp/tmp.comp'
    model = SpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=64)
    model = model.to(device)
    itr, model = load_checkpoint(args.snapshot, model, only_net=True)
    model.eval()
    model.update()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    vgg16 = models.vgg16(pretrained=True).to(device)
    vgg16.eval()
    resnet18 = models.resnet18(pretrained=True).to(device)
    resnet18.eval()

    conf = dict(model_type='vgg', arch=vgg16, layer_name='features_29')
    camera = GradCAM.from_config(**conf)

    loader_subset = torch.utils.data.DataLoader(
        ImagenetDataset(args.imagenet, transforms.Compose([  # ImagenetDataset
            transforms.Resize(280),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # hyperparameters
    total_itr = 3
    print_itr = 1000
    w = 10000
    grad_clip = 1e1
    lmbda = 0.01

    result = dict()

    # original image
    print('[Info] Original images')
    result_init(result, 'original')

    acc_1_avg = AverageMeter()
    acc_5_avg = AverageMeter()
    bpp_avg = AverageMeter()
    for x, label in loader_subset:
        x = x.to(device)
        label = label.to(device)
        with torch.no_grad():
            top1, top5 = eval_classification(resnet18, x, label)
        acc_1_avg.update(top1)
        acc_5_avg.update(top5)

    result['original']['acc1'].append(acc_1_avg.avg)
    result['original']['acc5'].append(acc_5_avg.avg)
    print(f'[Original] Acc@1: {acc_1_avg.avg:.4f} | Acc@5: {acc_5_avg.avg:.4f} ')

    # uniform qmap
    print('[Info] Uniform quality map')
    N = 11
    result_init(result, 'uniform')

    for q in range(N):
        Q = q / (N - 1)
        acc_1_avg = AverageMeter()
        acc_5_avg = AverageMeter()
        bpp_avg = AverageMeter()
        for x, label in loader_subset:
            x = x.to(device)
            label = label.to(device)
            with torch.no_grad():
                x_recon, bpp = recon_uniform(model, x, q=Q)
                top1, top5 = eval_classification(resnet18, x_recon, label)
            acc_1_avg.update(top1)
            acc_5_avg.update(top5)
            bpp_avg.update(bpp)

        result['uniform']['bpp'].append(bpp_avg.avg)
        result['uniform']['acc1'].append(acc_1_avg.avg)
        result['uniform']['acc5'].append(acc_5_avg.avg)
        print(f'[{Q:.1f}] BPP: {bpp_avg.avg:.4f} | Acc@1: {acc_1_avg.avg:.4f} | Acc@5: {acc_5_avg.avg:.4f} ')

    # gradcam as qmap
    print('[Info] Grad-CAM as quality map')
    N = 11
    result_init(result, 'cam')

    for q in range(N):
        alpha = q / (N - 1)
        acc_1_avg = AverageMeter()
        acc_5_avg = AverageMeter()
        bpp_avg = AverageMeter()
        for x, label in loader_subset:
            x = x.to(device)
            label = label.to(device)
            qmap = alpha * get_grad_cam(x).to(device)
            with torch.no_grad():
                x_hat_decoded, bpp = recon_with_qmap(model, x, qmap)
                top1, top5 = eval_classification(resnet18, x_hat_decoded, label)
            acc_1_avg.update(top1)
            acc_5_avg.update(top5)
            bpp_avg.update(bpp)

        result['cam']['bpp'].append(bpp_avg.avg)
        result['cam']['acc1'].append(acc_1_avg.avg)
        result['cam']['acc5'].append(acc_5_avg.avg)
        print(f'[{alpha:.1f}] BPP: {bpp_avg.avg:.4f} | Acc@1: {acc_1_avg.avg:.4f} | Acc@5: {acc_5_avg.avg:.4f} ')

    # optimizing qmap
    print('[Info] Optimized quality map')

    for total_itr in [3, 5]:  # 3, 5
        result_init(result, total_itr)
        for lmbda in [0.0001, 0.001, 0.004, 0.01, 0.1, 1, 10, 100, 1000]:
            acc_1_avg = AverageMeter()
            acc_5_avg = AverageMeter()
            bpp_avg = AverageMeter()
            time_avg = AverageMeter()
            for i, (x, label) in enumerate(loader_subset):  # loader  loader_subset
                x = x.to(device)
                label = label.to(device).long()[0]

                t_start = time()

                train_param = init(x, label, lmbda=lmbda)
                optimize(train_param, total_itr)

                t_end = time()

                qmap_norm_best = normalize_qmap(train_param['qmap_best'])

                x_hat_decoded, bpp = recon_with_qmap(model, x, qmap_norm_best)
                top1, top5 = eval_classification(resnet18, x_hat_decoded, label)
                acc_1_avg.update(top1)
                acc_5_avg.update(top5)
                bpp_avg.update(bpp)
                time_avg.update(t_end - t_start)

                if (i+1) % print_itr == 0:
                    print(f'[{total_itr}, {lmbda}, {i:>3}] | BPP: {bpp_avg.avg:.4f} | '
                          f'Acc@1: {acc_1_avg.avg:.4f} | Acc@5: {acc_5_avg.avg:.4f} | Optim. time: {time_avg.avg:.2f}s')

            print(f'[{total_itr}, {lmbda}] | BPP: {bpp_avg.avg:.4f} | '
                  f'Acc@1: {acc_1_avg.avg:.4f} | Acc@5: {acc_5_avg.avg:.4f} | Optim. time: {time_avg.avg:.2f}s')

            result[total_itr]['bpp'].append(bpp_avg.avg)
            result[total_itr]['acc1'].append(acc_1_avg.avg)
            result[total_itr]['acc5'].append(acc_5_avg.avg)
    plot(result)
