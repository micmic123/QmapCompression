import argparse
import sys
import os
from tqdm import tqdm

import torch
from compressai.zoo import models

from train import quality2lambda
from models.models import SpatiallyAdaptiveCompression
from dataset import get_dataloader, get_test_dataloader_compressai
from utils import load_checkpoint, AverageMeter, get_config, _encode, _decode
from losses.losses import Metrics, PixelwiseRateDistortionLoss


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Pixelwise Variable Rate Compression Evaluation')
    parser.add_argument('--testset', help='testset path', type=str, default='./data/kodak.csv')
    parser.add_argument('--level', help='', type=int, default=1)
    parser.add_argument(
        "--model",
        choices=models.keys(),
        default=list(models.keys())[0],
        help="NN model to use (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=list(range(1, 9)),
        type=int,
        default=3,
        help="Quality setting (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--metric",
        choices=["mse"],
        default="mse",
        help="metric trained against (default: %(default)s",
    )
    args = parser.parse_args(argv)
    return args


def test(test_dataloader, model, metric):
    device = next(model.parameters()).device

    with torch.no_grad():
        bpp_avg = AverageMeter()
        bpp_real_avg = AverageMeter()
        psnr_avg = AverageMeter()
        ms_ssim_avg = AverageMeter()
        enc_time_avg = AverageMeter()
        dec_time_avg = AverageMeter()

        for x, _ in tqdm(test_dataloader):
            x = x.to(device)
            out_net = model(x)

            bpp_real, out, enc_time = _encode(model, x, '/tmp/comp')

            x_hat_decoded, dec_time = _decode(model, '/tmp/comp', coder='ans', verbose=False)
            out_net['x_hat'] = x_hat_decoded
            bpp, psnr, ms_ssim = metric(out_net, x)

            bpp_avg.update(bpp.item())
            bpp_real_avg.update(bpp_real)
            psnr_avg.update(psnr.item())
            ms_ssim_avg.update(ms_ssim.item())
            enc_time_avg.update(enc_time)
            dec_time_avg.update(dec_time)

        print(
            f'[ Test ]'
            f' Real Bpp: {bpp_real_avg.avg:.4f} |'
            f' Bpp: {bpp_avg.avg:.4f} |'
            f' PSNR: {psnr_avg.avg:.4f} |'
            f' MS-SSIM: {ms_ssim_avg.avg:.4f} |'
            f' Enc Time: {enc_time_avg.avg:.4f}s |'
            f' Dec Time: {dec_time_avg.avg:.4f}s'
        )


def main(argv):
    args = parse_args(argv)
    config = {
        'batchsize_test': 1,
        'testset': args.testset
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = Metrics()
    test_dataloader = get_test_dataloader_compressai(config)

    model = models[args.model](quality=args.quality, metric=args.metric, pretrained=True)
    model = model.to(device)
    model.eval()
    model.update()
    test(test_dataloader, model, metric)


if __name__ == '__main__':
    main(sys.argv[1:])
