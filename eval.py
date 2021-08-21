import argparse
import sys
import os

import torch

from train import quality2lambda
from models.models import SpatiallyAdaptiveCompression
from dataset import get_dataloader
from utils import load_checkpoint, AverageMeter, get_config, _encode, _decode
from losses.losses import Metrics, PixelwiseRateDistortionLoss


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Pixelwise Variable Rate Compression Evaluation')
    parser.add_argument('--snapshot', help='snapshot path', type=str, required=True)
    parser.add_argument('--testset', help='testset path', type=str, default='./data/kodak.csv')
    parser.add_argument('--tqdm', help='use tqdm', action='store_true', default=False)
    parser.add_argument('--level', help='number of levels (real - 1)', type=int, default=10)
    args = parser.parse_args(argv)

    assert args.snapshot.startswith('./')
    dir_path = '/'.join(args.snapshot.split('/')[:-2])
    args.config = os.path.join(dir_path, 'config.yaml')

    return args


def test(test_dataloaders, model, criterion, metric):
    device = next(model.parameters()).device
    loss_all_avg = AverageMeter()
    enc_time_all_avg = AverageMeter()
    dec_time_all_avg = AverageMeter()

    with torch.no_grad():
        for i, test_dataloader in enumerate(test_dataloaders):
            loss_avg = AverageMeter()
            aux_loss_avg = AverageMeter()
            bpp_avg = AverageMeter()
            bpp_real_avg = AverageMeter()
            psnr_avg = AverageMeter()
            ms_ssim_avg = AverageMeter()
            enc_time_avg = AverageMeter()
            dec_time_avg = AverageMeter()

            for x, qmap in test_dataloader:
                x = x.to(device)
                qmap = qmap.to(device)
                lmbdamap = quality2lambda(qmap)
                out_net = model(x, qmap)

                bpp_real, out, enc_time = _encode(model, x, '/tmp/comp', qmap)

                x_hat_decoded, dec_time = _decode(model, '/tmp/comp', coder='ans', verbose=False)
                out_net['x_hat'] = x_hat_decoded
                out_criterion = criterion(out_net, x, lmbdamap)
                bpp, psnr, ms_ssim = metric(out_net, x)

                loss_avg.update(out_criterion['loss'].item())
                aux_loss_avg.update(model.aux_loss().item())
                bpp_avg.update(out_criterion['bpp_loss'].item())
                bpp_real_avg.update(bpp_real)
                psnr_avg.update(psnr.item())
                ms_ssim_avg.update(ms_ssim.item())
                enc_time_avg.update(enc_time)
                dec_time_avg.update(dec_time)

            level = i - 1
            print(
                f'[ Test{level:>2} ]'
                f' Total: {loss_avg.avg:.4f} |'
                f' Real BPP: {bpp_real_avg.avg:.4f} |'
                f' BPP: {bpp_avg.avg:.4f} |'
                f' PSNR: {psnr_avg.avg:.4f} |'
                f' MS-SSIM: {ms_ssim_avg.avg:.4f} |'
                f' Aux: {aux_loss_avg.avg:.4f} |'
                f' Enc Time: {enc_time_avg.avg:.4f}s |'
                f' Dec Time: {dec_time_avg.avg:.4f}s'
            )

            # uniform qmap
            if level != -1:
                loss_all_avg.update(loss_avg.avg)
            enc_time_all_avg.update(enc_time_avg.avg)
            dec_time_all_avg.update(dec_time_avg.avg)

        print(f'[ Test ] Total mean: {loss_all_avg.avg:.4f} |' 
              f' Enc Time: {enc_time_all_avg.avg:.4f}s |'
              f' Dec Time: {dec_time_all_avg.avg:.4f}s')


def main(argv):
    args = parse_args(argv)
    config = get_config(args.config)
    config['batchsize_test'] = 1
    config['testset'] = args.testset

    print('[config]', args.config)
    msg = f'======================= {args.snapshot} ======================='
    print(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p', 'testset'}:
            print(f' *{k}: ', v)
        else:
            print(f'  {k}: ', v)
    print('=' * len(msg))
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = Metrics()
    criterion = PixelwiseRateDistortionLoss()
    train_dataloader, test_dataloaders = get_dataloader(config, L=args.level)

    model = SpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=64)
    model = model.to(device)
    itr, model = load_checkpoint(args.snapshot, model, only_net=True)
    model.eval()
    model.update()
    test(test_dataloaders, model, criterion, metric)


if __name__ == '__main__':
    main(sys.argv[1:])
