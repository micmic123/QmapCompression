import argparse
import random
import sys
import os
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim

from models.models import SpatiallyAdaptiveCompression
from dataset import get_dataloader
from utils import init, Logger, load_checkpoint, save_checkpoint, AverageMeter
from losses.losses import Metrics, PixelwiseRateDistortionLoss


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Spatially-Adaptive Variable Rate Compression')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='result dir name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str)
    parser.add_argument('--resume', help='snapshot path', type=str)
    parser.add_argument('--seed', help='seed number', default=None, type=int)
    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './configs/config.yaml'

    return args


# T in the paper
def quality2lambda(qmap):
    return 1e-3 * torch.exp(4.382 * qmap)


def test(logger, test_dataloaders, model, criterion, metric):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for i, test_dataloader in enumerate(test_dataloaders):
            logger.init()
            for x, qmap in test_dataloader:
                x = x.to(device)
                qmap = qmap.to(device)
                lmbdamap = quality2lambda(qmap)
                out_net = model(x, qmap)
                out_net['x_hat'].clamp_(0, 1)

                out_criterion = criterion(out_net, x, lmbdamap)
                bpp, psnr, ms_ssim = metric(out_net, x)

                logger.update_test(bpp, psnr, ms_ssim, out_criterion, model.aux_loss())
            level = i-1
            logger.print_test(level)
            logger.write_test(level)
            if level != -1:
                # uniform qmap
                loss.update(logger.loss.avg)
                bpp_loss.update(logger.bpp_loss.avg)
                mse_loss.update(logger.mse_loss.avg)
        print(f'[ Test ] Total mean: {loss.avg:.4f}')
    logger.init()
    model.train()

    return loss.avg, bpp_loss.avg, mse_loss.avg


def train(args, config, base_dir, snapshot_dir, output_dir, log_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = PixelwiseRateDistortionLoss()
    metric = Metrics()
    train_dataloader, test_dataloaders = get_dataloader(config)
    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)

    model = SpatiallyAdaptiveCompression(N=config['N'], M=config['M'], sft_ks=config['sft_ks'], prior_nc=64)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    aux_optimizer = optim.Adam(model.aux_parameters(), lr=config['lr_aux'])

    if args.resume:
        itr, model = load_checkpoint(args.resume, model, optimizer, aux_optimizer)
        logger.load_itr(itr)

    if config['set_lr']:
        lr_prior = optimizer.param_groups[0]['lr']
        for g in optimizer.param_groups:
            g['lr'] = float(config['set_lr'])
        print(f'[set lr] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

    model.train()
    loss_best = 1e10
    while logger.itr < config['max_itr']:
        for x, qmap in train_dataloader:
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            x = x.to(device)
            qmap = qmap.to(device)
            lmbdamap = quality2lambda(qmap)

            out_net = model(x, qmap)
            out_criterion = criterion(out_net, x, lmbdamap)

            out_criterion['loss'].backward()
            aux_loss = model.aux_loss()
            aux_loss.backward()

            # for stability
            if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion['loss'] > 10000:
                continue

            if config['clip_max_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_max_norm'])
            optimizer.step()
            aux_optimizer.step()  # update quantiles of entropy bottleneck modules

            # logging
            logger.update(out_criterion, aux_loss)
            if logger.itr % config['log_itr'] == 0:
                logger.print()
                logger.write()
                logger.init()

            # test and save model snapshot
            if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
                model.update()
                loss, bpp_loss, mse_loss = test(logger, test_dataloaders, model, criterion, metric)
                if loss < loss_best:
                    print('Best!')
                    save_checkpoint(os.path.join(snapshot_dir, 'best.pt'), logger.itr, model, optimizer, aux_optimizer)
                    loss_best = loss
                if logger.itr % config['snapshot_save_itr'] == 0:
                    save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:07}_{bpp_loss:.4f}_{mse_loss:.8f}.pt'),
                                    logger.itr, model, optimizer, aux_optimizer)

            # lr scheduling
            if logger.itr % config['lr_shedule_step'] == 0:
                lr_prior = optimizer.param_groups[0]['lr']
                for g in optimizer.param_groups:
                    g['lr'] *= config['lr_shedule_scale']
                print(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')


def main(argv):
    args = parse_args(argv)
    config, base_dir, snapshot_dir, output_dir, log_dir = init(args)
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        # torch.backends.cudnn.deterministic = True  # slow
        # torch.backends.cudnn.benchmark = False

    print('[PID]', os.getpid())
    print('[config]', args.config)
    msg = f'======================= {args.name} ======================='
    print(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p'}:
            print(f' *{k}: ', v)
        else:
            print(f'  {k}: ', v)
    print('=' * len(msg))
    print()

    train(args, config, base_dir, snapshot_dir, output_dir, log_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
