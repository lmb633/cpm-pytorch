import argparse
import torch
import torch.nn as nn
import numpy as np
from data_gen import lsp_data
from torch.utils.data import DataLoader
import os
import time
from model_old import CPM
from utils import AverageMeter, save_checkpoint, device, visualize, adjust_learning_rate
import Mytransforms
from lsp_data import LSP_Data


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end_epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=4e-6, help='start learning rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in each context')
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

    train_set = lsp_data()
    # train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    train_loader = DataLoader(LSP_Data(), batch_size=args.batch_size, shuffle=True)
    if not os.path.exists(checkpoint_path):
        print('========train from beginning==========')
        model = CPM()
        model = torch.nn.DataParallel(model).to(device)
        if args.optimizer == 'sgd':
            print('=========use SGD=========')
            optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            print('=========use ADAM=========')
            optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('=========load checkpoint============')
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        optimizer = checkpoint['optimizer']
        best_loss = checkpoint['best_loss']
        print('epoch: ', start_epoch, 'best_loss: ', best_loss)
    criterion = nn.MSELoss().to(device)
    for epoch in range(start_epoch, args.end_epoch):
        loss = train_once(train_loader, model, criterion, optimizer, args)
        print('==== avg lose of epoch {0} is {1} ====='.format(epoch, loss))
        if loss < best_loss:
            print('============= loss down =============')
            best_loss = loss
            epochs_since_improvement = 0
            save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss)
        else:
            print('============== loss not improvement ============ ')
            epochs_since_improvement += 1
            # visualize(model)


heat_weight = 46 * 46 * 15 / 1.0


def train_once(trainloader, model, criterion, optimizer, args):
    losses = AverageMeter()
    model.train()
    for i, (img, heatmap, centermap) in enumerate(trainloader):
        img = img.to(device)
        heatmap = heatmap.to(device)
        centermap = centermap.to(device)

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
        if i % args.print_freq == 0:
            print(time.asctime(), loss)

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    train(args)
