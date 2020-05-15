# -*-coding:UTF-8-*-
import argparse
import sys
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim

sys.path.append("..")
from utils import AverageMeter
import models
from lsp_data import LSP_Data
import os

batch_size = 32


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default='BEST_checkpoint.tar', type=str,
                        dest='pretrained', help='the path of pretrained model')
    return parser.parse_args()


def construct_model(args):
    model = models.CPM(k=14)
    model = torch.nn.DataParallel(model).cuda()
    if os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained)['state_dict']
        model.load_state_dict(state_dict)
    return model


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


def train_val(model):
    cudnn.benchmark = True
    # train
    train_loader = torch.utils.data.DataLoader(
        LSP_Data(),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    params, multiple = get_parameters(model, False)
    optimizer = torch.optim.SGD(params, 1e-5, momentum=0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    iters = 0

    heat_weight = 46 * 46 * 15 / 1.0

    while iters < 1000000:

        for i, (input, heatmap, centermap, _) in enumerate(train_loader):

            data_time.update(time.time() - end)

            heatmap = heatmap.cuda(async=True)
            centermap = centermap.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)
            centermap_var = torch.autograd.Variable(centermap)

            heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, centermap_var)

            loss1 = criterion(heat1, heatmap_var) * heat_weight
            loss2 = criterion(heat2, heatmap_var) * heat_weight
            loss3 = criterion(heat3, heatmap_var) * heat_weight
            loss4 = criterion(heat4, heatmap_var) * heat_weight
            loss5 = criterion(heat5, heatmap_var) * heat_weight
            loss6 = criterion(heat6, heatmap_var) * heat_weight

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            losses.update(loss.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            iters += 1
            if iters % 100 == 0:
                print('Train Iteration: {0}\t'
                      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, loss=losses))
                print(time.strftime(
                    '%Y-%m-%d %H:%M:%S ----------------------------------------\n', time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                save_checkpoint({'iter': iters, 'state_dict': model.state_dict(), })


def save_checkpoint(state):
    torch.save(state, 'BEST_checkpoint.tar')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse()
    model = construct_model(args)
    train_val(model)
