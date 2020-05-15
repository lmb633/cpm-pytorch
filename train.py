import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_gen import lsp_data
from models import CPM
from utils import AverageMeter, save_checkpoint, device, visualize, adjust_learning_rate

heat_weight = 46 * 46 * 15 / 1.0


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end_epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=4e-6, help='start learning rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.0, help='momentum')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default='BEST_checkpoint.tar', help='checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='checkpoint')
    parser.add_argument('--shrink_factor', type=float, default=0.5, help='checkpoint')
    args = parser.parse_args()
    return args


def train(args):
    # torch.manual_seed(3)
    # np.random.seed(3)
    checkpoint_path = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    train_loader = DataLoader(lsp_data(), batch_size=args.batch_size, shuffle=True)
    model = CPM(k=14)
    model = torch.nn.DataParallel(model).cuda()
    if os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained)['model']
        model.load_state_dict(state_dict)
        print('epoch: ', start_epoch, 'best_loss: ', best_loss)
    params, multiple = get_parameters(model, False)
    optimizer = torch.optim.SGD(params, 1e-5, momentum=0)
    # if args.optimizer == 'sgd':
    #     print('=========use SGD=========')
    # optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
    # else:
    #     print('=========use ADAM=========')
    #     optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss().cuda()

    while start_epoch < args.end_epoch:
        # if epochs_since_improvement == 10:
        #     break
        # if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
        #     print('============= reload model ,adjust lr ===============')
        #     checkpoint = torch.load(checkpoint_path)
        #     model = CPM()
        #     model = torch.nn.DataParallel(model).to(device)
        #     model.load_state_dict(checkpoint['model'])
        #     optimizer = checkpoint['optimizer']
        #     best_loss = checkpoint['best_loss']
        #     adjust_learning_rate(optimizer, args.shrink_factor)
        # model = torch.nn.DataParallel(model).to(device)
        # loss = train_once(train_loader, model, criterion, optimizer, epoch, args)
        losses = AverageMeter()
        for i, (img, heatmap, centermap, _) in enumerate(train_loader):
            img = img.to(device)
            heatmap = heatmap.cuda(async=True)
            centermap = centermap.cuda(async=True)

            img = torch.autograd.Variable(img)
            heatmap = torch.autograd.Variable(heatmap)
            centermap = torch.autograd.Variable(centermap)

            heatmap1, heatmap2, heatmap3, heatmap4, heatmap5, heatmap6 = model(img, centermap)
            loss1 = criterion(heatmap1, heatmap) * heat_weight
            loss2 = criterion(heatmap2, heatmap) * heat_weight
            loss3 = criterion(heatmap3, heatmap) * heat_weight
            loss4 = criterion(heatmap4, heatmap) * heat_weight
            loss5 = criterion(heatmap5, heatmap) * heat_weight
            loss6 = criterion(heatmap6, heatmap) * heat_weight

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), img.size(0))
            start_epoch += 1
            if i % args.print_freq == 0:
                print(time.asctime(), loss)
                print('epoch: {0} iter: {1}/{2} loss: {loss.val:.4f}({loss.avg:.4f})'.format(start_epoch, i, len(train_loader), loss=losses))
                save_checkpoint(start_epoch, epochs_since_improvement, model, optimizer, loss)

                # print('==== avg lose of epoch {0} is {1} ====='.format(epoch, loss))
                # if loss < best_loss:
                #     print('============= loss down =============')
                #     best_loss = loss
                #     epochs_since_improvement = 0
                # save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss)
                # else:
                #     print('============== loss not improvement ============ ')
                #     epochs_since_improvement += 1
                #     # visualize(model)


def get_parameters(model, isdefault=True):
    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': 1e-5},
              {'params': lr_2, 'lr': 1e-5 * 2.},
              {'params': lr_4, 'lr': 1e-5 * 4.},
              {'params': lr_8, 'lr': 1e-5 * 8.}]

    return params, [1., 2., 4., 8.]


def train_once(trainloader, model, criterion, optimizer, epoch, args):
    model.train()
    losses = AverageMeter()
    for i, (img, heatmap, centermap, _) in enumerate(trainloader):
        img = img.to(device)
        heatmap = heatmap.to(device)
        centermap = centermap.to(device)

        img = torch.autograd.Variable(img)
        heatmap = torch.autograd.Variable(heatmap)
        centermap = torch.autograd.Variable(centermap)
        # mask = mask.to(device).unsqueeze(dim=2).unsqueeze(dim=3)

        heatmap1, heatmap2, heatmap3, heatmap4, heatmap5, heatmap6 = model(img, centermap)
        # loss1 = criterion(heatmap1 * mask, heatmap * mask) * heat_weight
        # loss2 = criterion(heatmap2 * mask, heatmap * mask) * heat_weight
        # loss3 = criterion(heatmap3 * mask, heatmap * mask) * heat_weight
        # loss4 = criterion(heatmap4 * mask, heatmap * mask) * heat_weight
        # loss5 = criterion(heatmap5 * mask, heatmap * mask) * heat_weight
        # loss6 = criterion(heatmap6 * mask, heatmap * mask) * heat_weight

        loss1 = criterion(heatmap1, heatmap) * heat_weight
        loss2 = criterion(heatmap2, heatmap) * heat_weight
        loss3 = criterion(heatmap3, heatmap) * heat_weight
        loss4 = criterion(heatmap4, heatmap) * heat_weight
        loss5 = criterion(heatmap5, heatmap) * heat_weight
        loss6 = criterion(heatmap6, heatmap) * heat_weight

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img.size(0))

        if i % args.print_freq == 0:
            print(time.asctime(), loss)
            print('epoch: {0} iter: {1}/{2} loss: {loss.val:.4f}({loss.avg:.4f})'.format(epoch, i, len(trainloader), loss=losses))
    print(loss)
    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    train(args)
