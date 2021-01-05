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
from torch.utils.tensorboard import SummaryWriter

import resnet
import cifar
from utils.statistics_meter import AverageMeter


model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(
    description='Propert ResNets for CIFAR10 in pytorch',
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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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
parser.add_argument('--print-freq', '-p', default=300, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--log-freq', '--lf', default=100, type=int, metavar='N',
                    help="TensorBoard log frequency during training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision (16-bit)')
parser.add_argument('--use-lr-warmup', action='store_true',
                    help="Use learning scheduler 2 to warmup the learning rate")
parser.add_argument('--comment', type=str, help='Commentary on the run')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    cudnn.benchmark = True

    dataset = cifar.__dict__[args.dataset]('~/datasets', pin_memory=True)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    comments = "_".join([x[1] for x in [(args.use_lr_warmup, "use_lr_warmup"), (args.comment, args.comment)] if x[0]])

    writer = SummaryWriter(log_dir=os.path.join(
        'runs',
        '{current_time}_{hostname}_{net}_gpu{gpus}_wd={weight_decay}_lr={lr}_b={bs}_j{num_workers}{comments}'.format(
            current_time=datetime.now().strftime('%b%d_%H-%M-%S'),
            hostname=socket.gethostname(),
            net=args.arch,
            gpus=os.environ['CUDA_VISIBLE_DEVICES'],
            weight_decay=args.weight_decay,
            lr=args.lr,
            bs=args.batch_size,
            num_workers=args.workers,
            comments=("_" + comments) if comments else ""
        )
    ))

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

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
    if args.use_lr_warmup:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            gamma=10,
            milestones=[2] # First two epochs
        )
    else:
        lr_scheduler2 = None

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, writer)
        lr_scheduler.step()
        if lr_scheduler2:
            lr_scheduler2.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)

    # TODO: add precision-recall curve
    if hasattr(writer, "flush"):
        writer.flush()
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

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

        if i % args.print_freq == (args.print_freq-1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'DL {data_time.val:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        if i % args.log_freq == (args.log_freq-1):
            writer.add_scalar("Prec1 train", top1.avg, epoch + i/len(train_loader))
            writer.add_scalar("Train loss", losses.avg, epoch + i/len(train_loader))

    writer.add_scalar("Prec1 train", top1.avg, epoch+1)
    writer.add_scalar("Train loss", losses.avg, epoch+1)
    print(f"Train: [{epoch}]\t\t"
          f"Time {batch_time.avg:.3f}\t"
          f"(DL {data_time.avg:.3f})\t"
          f"Loss {losses.avg:.4f}\t\t"
          f"Prec@1 {top1.avg:.3f}")


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

        writer.add_scalar("Prec1 valid", top1.avg, epoch)
        writer.add_scalar("Valid loss", top1.avg, epoch)

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
