"""
main.py  —  ProtoPFormer adapted for LIDC-IDRI lung-nodule binary classification
=================================================================================
Usage:
    python main.py \
        --data_path datasets/LIDC \
        --lidc_format folder \
        --base_architecture deit_tiny_patch16_224 \
        --prototype_shape 20 192 1 1 \
        --use_global True \
        --use_ppc_loss True \
        --epochs 100 \
        --batch_size 32 \
        --output_dir output/LIDC/run1
"""

import argparse
import datetime
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, ModelEma, get_state_dict

import protopformer
import tools.utils as utils
from tools.create_optimizer  import create_optimizer
from tools.create_scheduler  import create_scheduler
from tools.datasets          import build_dataset
from tools.engine_proto      import (train_one_epoch, evaluate,
                                     compute_class_weights)
from tools.utils             import str2bool


# ── argument parser ────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser('ProtoPFormer LIDC training', add_help=False)

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument('--data_path',    type=str,
                   default='datasets/LIDC')
    p.add_argument('--lidc_format',  type=str, default='folder',
                   choices=['folder', 'csv'],
                   help="'folder' = class-named subdirs; 'csv' = manifest file")
    p.add_argument('--img_size',     type=int, default=224)

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument('--base_architecture', type=str,
                   default='deit_tiny_patch16_224',
                   choices=['deit_tiny_patch16_224',
                             'deit_small_patch16_224',
                             'cait_xxs24_224'])
    p.add_argument('--prototype_shape', nargs='+', type=int,
                   default=[20, 192, 1, 1],
                   help='[num_prototypes, dim, 1, 1] — num_prototypes must be '
                        'even (equal split between benign / malignant)')
    p.add_argument('--prototype_activation_function', type=str, default='log')
    p.add_argument('--add_on_layers_type', type=str, default='regular')

    # ── global branch ─────────────────────────────────────────────────────────
    p.add_argument('--use_global',            type=str2bool, default=True)
    p.add_argument('--global_proto_per_class', type=int,     default=5)
    p.add_argument('--global_coe',             type=float,   default=0.3)

    # ── local branch / token pruning ──────────────────────────────────────────
    p.add_argument('--reserve_layers',      nargs='+', type=int, default=[11])
    p.add_argument('--reserve_token_nums',  nargs='+', type=int, default=[81])

    # ── PPC loss ──────────────────────────────────────────────────────────────
    p.add_argument('--use_ppc_loss',   type=str2bool, default=True)
    p.add_argument('--ppc_cov_thresh', type=float,    default=2.)
    p.add_argument('--ppc_mean_thresh',type=float,    default=2.)
    p.add_argument('--ppc_cov_coe',    type=float,    default=0.1)
    p.add_argument('--ppc_mean_coe',   type=float,    default=0.5)

    # ── loss coefficients ─────────────────────────────────────────────────────
    p.add_argument('--use_class_weights', type=str2bool, default=True,
                   help='Inverse-frequency class weighting for imbalanced data')
    p.add_argument('--coefs_crs_ent', type=float, default=1.0)
    p.add_argument('--coefs_clst',    type=float, default=0.8)
    p.add_argument('--coefs_sep',     type=float, default=-0.08)
    p.add_argument('--coefs_l1',      type=float, default=1e-4)

    # ── optimiser ─────────────────────────────────────────────────────────────
    p.add_argument('--opt',            type=str,   default='adamw')
    p.add_argument('--opt-eps',        type=float, default=1e-8)
    p.add_argument('--opt-betas',      type=float, nargs='+', default=None)
    p.add_argument('--clip_grad',      type=float, default=None)
    p.add_argument('--momentum',       type=float, default=0.9)
    p.add_argument('--weight_decay',   type=float, default=0.05)
    p.add_argument('--features_lr',         type=float, default=1e-4)
    p.add_argument('--add_on_layers_lr',    type=float, default=3e-3)
    p.add_argument('--prototype_vectors_lr',type=float, default=3e-3)

    # ── LR schedule ───────────────────────────────────────────────────────────
    p.add_argument('--sched',           type=str,   default='cosine')
    p.add_argument('--lr',              type=float, default=5e-4)
    p.add_argument('--warmup-lr',       type=float, default=1e-6)
    p.add_argument('--min-lr',          type=float, default=1e-5)
    p.add_argument('--decay-epochs',    type=float, default=30)
    p.add_argument('--warmup-epochs',   type=int,   default=5)
    p.add_argument('--cooldown-epochs', type=int,   default=10)
    p.add_argument('--patience-epochs', type=int,   default=10)
    p.add_argument('--decay-rate',      type=float, default=0.1)

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument('--epochs',        type=int, default=100)
    p.add_argument('--batch_size',    type=int, default=32)
    p.add_argument('--save_ep_freq',  type=int, default=50)
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--num_workers',   type=int, default=4)
    p.add_argument('--pin-mem',       action='store_true', default=True)
    p.add_argument('--device',        type=str, default='cuda')
    p.add_argument('--output_dir',    type=str, default='output/LIDC/run1')
    p.add_argument('--resume',        type=str, default='')
    p.add_argument('--start_epoch',   type=int, default=0)
    p.add_argument('--eval',          action='store_true')

    # ── EMA ───────────────────────────────────────────────────────────────────
    p.add_argument('--model_ema',           action='store_true',  default=True)
    p.add_argument('--model-ema-decay',     type=float, default=0.99996)
    p.add_argument('--model-ema-force-cpu', action='store_true',  default=False)

    # ── distributed ───────────────────────────────────────────────────────────
    p.add_argument('--world_size',  type=int, default=1)
    p.add_argument('--dist_url',    type=str, default='env://')
    p.add_argument('--dist-eval',   action='store_true', default=False)

    # ── label smoothing / mixup (off by default for medical imaging) ──────────
    p.add_argument('--smoothing',       type=float, default=0.0)
    p.add_argument('--enable_smoothing',type=bool,  default=False)
    p.add_argument('--enable_mixup',    type=bool,  default=False)
    p.add_argument('--mixup',           type=float, default=0.0)
    p.add_argument('--cutmix',          type=float, default=0.0)
    p.add_argument('--cutmix-minmax',   type=float, nargs='+', default=None)
    p.add_argument('--mixup-prob',      type=float, default=1.0)
    p.add_argument('--mixup-switch-prob',type=float,default=0.5)
    p.add_argument('--mixup-mode',      type=str,   default='batch')

    return p


# ── helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_logger_and_writer(args):
    if args.eval:
        log_dir = os.path.join(args.output_dir, 'eval-logs')
    else:
        log_dir = os.path.join(args.output_dir, 'train-logs')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    tb_dir   = os.path.join(args.output_dir, 'tf-logs')
    for d in [log_dir, ckpt_dir, tb_dir]:
        os.makedirs(d, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=1)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='w'),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger('lidc_train')
    return tb_writer, logger


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    utils.init_distributed_mode(args)
    tb_writer, logger = get_logger_and_writer(args)
    device = torch.device(args.device)

    logger.info(f"Arguments:\n{args}")

    # ── datasets ──────────────────────────────────────────────────────────────
    dataset_train, nb_classes = build_dataset(is_train=True,  args=args)
    dataset_val,   _          = build_dataset(is_train=False, args=args)
    assert nb_classes == 2

    logger.info(f"Train: {len(dataset_train)} samples  |  Val: {len(dataset_val)} samples")

    if hasattr(dataset_train, 'class_counts'):
        logger.info(f"Train class counts: {dataset_train.class_counts}")

    # ── data loaders ──────────────────────────────────────────────────────────
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train,
                                                             shuffle=True)
        sampler_val   = (torch.utils.data.DistributedSampler(dataset_val,
                                                              shuffle=False)
                         if args.dist_eval
                         else torch.utils.data.SequentialSampler(dataset_val))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ── class-weighted loss ────────────────────────────────────────────────────
    if args.use_class_weights:
        class_weights = compute_class_weights(dataset_train, device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using class-weighted CrossEntropyLoss")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ── model ─────────────────────────────────────────────────────────────────
    model = protopformer.construct_PPNet(
        base_architecture           = args.base_architecture,
        pretrained                  = True,
        img_size                    = args.img_size,
        prototype_shape             = args.prototype_shape,
        num_classes                 = 2,
        reserve_layers              = args.reserve_layers,
        reserve_token_nums          = args.reserve_token_nums,
        use_global                  = args.use_global,
        use_ppc_loss                = args.use_ppc_loss,
        ppc_cov_thresh              = args.ppc_cov_thresh,
        ppc_mean_thresh             = args.ppc_mean_thresh,
        global_coe                  = args.global_coe,
        global_proto_per_class      = args.global_proto_per_class,
        prototype_activation_function = args.prototype_activation_function,
        add_on_layers_type          = args.add_on_layers_type,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model,
                             decay=args.model_ema_decay,
                             device='cpu' if args.model_ema_force_cpu else '')

    # ── optimiser ─────────────────────────────────────────────────────────────
    joint_lr = {
        'features':         args.features_lr,
        'add_on_layers':    args.add_on_layers_lr,
        'prototype_vectors':args.prototype_vectors_lr,
    }
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer    = create_optimizer(args, model_without_ddp,
                                    joint_optimizer_lrs=joint_lr)
    loss_scaler  = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ── resume ─────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        if not args.eval and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            args.start_epoch = ckpt['epoch'] + 1
        if args.model_ema and 'model_ema' in ckpt:
            utils._load_checkpoint_for_ema(model_ema, ckpt['model_ema'])

    # ── eval only ─────────────────────────────────────────────────────────────
    if args.eval:
        stats = evaluate(loader_val, model, device, args)
        logger.info(f"Eval results:\n"
                    f"  Acc={stats['acc1']:.2f}%  AUC={stats['auc']:.2f}%  "
                    f"Sensitivity={stats['sensitivity']:.2f}%  "
                    f"Specificity={stats['specificity']:.2f}%  "
                    f"Balanced-Acc={stats['balanced_acc']:.2f}%")
        return

    # ── training loop ──────────────────────────────────────────────────────────
    logger.info(f"Starting training for {args.epochs} epochs")
    best_auc      = 0.0
    best_acc      = 0.0
    output_dir    = Path(args.output_dir)
    start_time    = time.time()
    _it           = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, criterion=criterion,
            data_loader=loader_train,
            optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, max_norm=args.clip_grad,
            model_ema=model_ema, mixup_fn=None,
            args=args, tb_writer=tb_writer, iteration=_it,
        )
        _it += len(loader_train)
        lr_scheduler.step(epoch)

        val_stats = evaluate(loader_val, model, device, args)

        # ── logging ───────────────────────────────────────────────────────────
        logger.info(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_stats['loss']:.4f}  "
            f"val_acc={val_stats['acc1']:.2f}%  "
            f"val_auc={val_stats['auc']:.2f}%  "
            f"sensitivity={val_stats['sensitivity']:.2f}%  "
            f"specificity={val_stats['specificity']:.2f}%"
        )
        for k, v in val_stats.items():
            tb_writer.add_scalar(f'val/{k}', v, epoch)
        tb_writer.add_scalar('train/loss', train_stats['loss'], epoch)

        # ── periodic checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % args.save_ep_freq == 0:
            _save(output_dir / f'checkpoints/checkpoint-{epoch}.pth',
                  model_without_ddp, optimizer, lr_scheduler,
                  epoch, model_ema, loss_scaler, args)

        # ── best-AUC checkpoint (primary metric for medical imaging) ──────────
        if val_stats['auc'] > best_auc:
            best_auc = val_stats['auc']
            _save(output_dir / 'checkpoints/best-auc.pth',
                  model_without_ddp, optimizer, lr_scheduler,
                  epoch, model_ema, loss_scaler, args)
            logger.info(f"  ↑ New best AUC: {best_auc:.2f}%  (saved best-auc.pth)")

        # ── best-acc checkpoint ────────────────────────────────────────────────
        if val_stats['acc1'] > best_acc:
            best_acc = val_stats['acc1']
            _save(output_dir / 'checkpoints/best-acc.pth',
                  model_without_ddp, optimizer, lr_scheduler,
                  epoch, model_ema, loss_scaler, args)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training complete in {elapsed}")
    logger.info(f"Best AUC: {best_auc:.2f}%  |  Best Acc: {best_acc:.2f}%")


def _save(path, model, optimizer, lr_scheduler, epoch,
          model_ema, loss_scaler, args):
    utils.save_on_master({
        'model':        model.state_dict(),
        'optimizer':    optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch':        epoch,
        'model_ema':    get_state_dict(model_ema) if model_ema else None,
        'scaler':       loss_scaler.state_dict(),
        'args':         args,
    }, path)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args   = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
