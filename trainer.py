import argparse
from datetime import datetime
import os
import shutil
import socket
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler as topt_lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.tensorboard import SummaryWriter

import resnet
import cifar
from utils.statistics_meter import AverageMeter


model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Proper ResNets for CIFAR10 in pytorch',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--dataset', '--ds', default='CIFAR10',
                    choices=["CIFAR10", "CIFAR100", "CIFAR100Coarse"],
                    help="Dataset to use")
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--base-width', metavar='WIDTH', default=16, type=int,
                    help='width of the base layer')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate'
                    '\nNote that for ResNet-112/1202 it is 1e-2')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=2, type=int,
                    metavar='N', help='print frequency (per epoch)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--log-freq', '--lf', default=4, type=int, metavar='N',
                    help="TensorBoard log frequency during training (per epoch)")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision (16-bit)')
parser.add_argument('--use-lr-warmup', action='store_true',
                    help="Use learning scheduler 2 to warmup the learning rate")
parser.add_argument('--lr-warmup-num-epochs', type=int, default=2,
                    help='Number of epochs for the warmup, if set')
parser.add_argument('--comment', type=str, help='Commentary on the run')

FOLDER_INCLUDED_ARGS = [('ds', 'dataset'), ('bs', 'batch_size'), ('lr', 'lr'), ('wd', 'weight_decay')]
FOLDER_IGNORED_ARGS = ['arch', 'workers', 'resume', 'log_freq', 'print_freq', 'momentum', 'start_epoch', 'epochs']

best_prec1 = 0

ROOT_LOG_FOLDER = 'runs_tst'


class LRSchedulerSequence(LRScheduler):
    def __init__(self, *args):
        self.schedulers = []
        for scheduler in args:
            if isinstance(scheduler, LRScheduler):
                self.schedulers.append(scheduler)

    def step(self, *args, **kwargs):
        for scheduler in self.schedulers:
            scheduler.step(*args, **kwargs)

    def add_scheduler(self, *args):
        for scheduler in args:
            self.schedulers.append(scheduler)


def get_folder_name():
    global args
    attrs = ['{}'.format(datetime.now().strftime('%b%d_%H-%M-%S'))]
    attrs.append('{}'.format(socket.gethostname()))
    attrs.append('gpu{}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'all')))
    attrs.append(args.arch)
    arg_keys = sorted(vars(args).keys())
    for (arg_key_print, arg_key_name) in FOLDER_INCLUDED_ARGS:
        attrs.append(f'{arg_key_print}={getattr(args, arg_key_name)}')
        arg_keys.remove(arg_key_name)
    for arg_key in arg_keys:
        arg_val = getattr(args, arg_key)
        if arg_key in FOLDER_IGNORED_ARGS:
            pass
        elif arg_key == 'comment':
            pass
        elif arg_key == 'lr_warmup_num_epochs' and not args.use_lr_warmup:
            pass
        elif isinstance(arg_val, bool):
            if arg_val:
                attrs.append(arg_key)
        else:
            attrs.append('{}={}'.format(arg_key, arg_val))
    if args.comment:
        attrs.append(args.comment)
    return '_'.join(attrs)


def main():
    global args, best_prec1
    args = parser.parse_args()

    log_subfolder = get_folder_name()

    cudnn.benchmark = True

    dataset = cifar.__dict__[args.dataset]('~/datasets', pin_memory=True)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    writer = SummaryWriter(log_dir=os.path.join(
        ROOT_LOG_FOLDER, log_subfolder
    ))

    layout = {
        'Prec@1': {
            'prec@1': ['Multiline', ['Prec1/train', 'Prec1/valid']],
        },
        'Losses': {
            'Train': ['Multiline', ['Loss/train']],
            'Valid': ['Multiline', ['Loss/valid']]
        }
    }
    writer.add_custom_scalars(layout)

    train_loader = dataset.get_train_loader(args.batch_size, shuffle=True,
                                            num_workers=args.workers)

    val_loader = dataset.get_test_loader(128, num_workers=args.workers)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch](
        num_classes=dataset.get_num_classes(),
        base_width=args.base_width
    ))
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler1 = topt_lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[100, 150], last_epoch=args.start_epoch - 1)
    main_lr_scheduler = LRSchedulerSequence(lr_scheduler1)
    if args.use_lr_warmup:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
        lr_scheduler2 = topt_lr_scheduler.MultiStepLR(
            optimizer,
            gamma=10,
            milestones=[args.lr_warmup_num_epochs] # First two epochs
        )
        main_lr_scheduler.add_scheduler(lr_scheduler2)

    if args.print_freq < 1:
        args.print_freq = 1
    if args.log_freq < 1:
        args.log_freq = 1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: add hparams to TensorBoard

    train(train_loader, val_loader, model, criterion, optimizer, main_lr_scheduler, writer)

    # TODO: add precision-recall curve
    if hasattr(writer, "flush"):
        writer.flush()
    writer.close()


def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, writer):

    global best_prec1
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        writer.add_scalar("base_learning_rate", optimizer.param_groups[0]['lr'], epoch)
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, writer)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, writer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    print_period = (len(train_loader) // args.print_freq) + 1
    log_period = (len(train_loader) // args.log_freq) + 1

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        target = target.cuda()
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_period == print_period-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'DL {data_time.val:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        if i % log_period == log_period-1:
            writer.add_scalar("Prec1/train", top1.avg, epoch + i/len(train_loader))
            writer.add_scalar("Loss/train", losses.avg, epoch + i/len(train_loader))

    print('Epoch: [{0}][done]\t'
          'Time {batch_time.val:.3f}\t'
          'DL {data_time.val:.3f}\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              epoch, batch_time=batch_time,
              data_time=data_time, loss=losses, top1=top1))
    writer.add_scalar("Prec1/train", top1.avg, epoch+1)
    writer.add_scalar("Loss/train", losses.avg, epoch+1)


def validate(val_loader, model, criterion, epoch, writer):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            targets = targets.cuda()
            inputs = inputs.cuda()

            if args.half:
                inputs = inputs.half()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        writer.add_scalar("Prec1/valid", top1.avg, epoch)
        writer.add_scalar("Loss/valid", losses.avg, epoch)

    print(f"Valid: Prec1 {top1.avg:.3f} \t (Time: {batch_time.avg:.3f}, Loss: {losses.avg:.4f})")

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
