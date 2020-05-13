# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys

sys.path.append("..")
from utils import adjust_learning_rate, AverageMeter
import models
import lsp_data
import Mytransforms

import shutil

batch_size = 16


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default='BEST_checkpoint.tar', type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--model_name', default='../ckpt/cpm', type=str,
                        help='model name to save parameters')

    return parser.parse_args()


def construct_model(args):
    model = models.CPM(k=14)
    # load pretrained model
    import os
    if os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained)['state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]
        #     new_state_dict[name] = v
        model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).cuda()

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


def train_val(model, args):
    train_dir = args.train_dir
    val_dir = args.val_dir

    cudnn.benchmark = True

    # train
    train_loader = torch.utils.data.DataLoader(
        lsp_data.LSP_Data(),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    criterion = nn.MSELoss().cuda()

    params, multiple = get_parameters(model, False)

    optimizer = torch.optim.SGD(params, 1e-5, momentum=0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(6)]
    end = time.time()
    iters = 0

    heat_weight = 46 * 46 * 15 / 1.0

    while iters < 10000:

        for i, (input, heatmap, centermap) in enumerate(train_loader):

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
                    '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(6):
                    losses_list[cnt].reset()

                save_checkpoint({
                    'iter': iters,
                    'state_dict': model.state_dict(),
                })


def save_checkpoint(state):
    torch.save(state, 'BEST_checkpoint.tar')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse()
    model = construct_model(args)
    train_val(model, args)
