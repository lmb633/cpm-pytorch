import math
import os
import random

import cv2
import numpy as np
import scipy
import torch

from data_gen import path
from data_gen import guassian_kernel
from data_gen import mat_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss):
    print('=========== save checkpoint ============')
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'best_loss': best_loss,
             'model': model.state_dict(),
             'optimizer': optimizer}
    torch.save(state, 'BEST_checkpoint.tar')


def print_model(model, print_freq=10):
    paras = model.named_parameters()
    for i, (name, para) in enumerate(paras):
        if i % print_freq == 0:
            print(name, para.sum(), para[0].sum(), para[-1].sum())


# def visualize(model,)
#

def get_kpts(maps, img_h=368.0, img_w=368.0):
    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[1:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x, y])
    return kpts


def draw_paint(img_path, kpts):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255]]
    limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10, 11], [7, 6], [12, 3], [12, 2], [2, 1], [1, 0], [3, 4],
               [4, 5]]
    # img = transforms.ToPILImage()(img[0].cpu())
    # im = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    im = cv2.imread(img_path)
    im = cv2.resize(im, (368, 368))
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)
    global idx
    cv2.imwrite('visualize/{0}.jpg'.format(idx), im)
    idx += 1


# def test_example(model, sample):
#     # img = cv2.imread(img_path)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = cv2.resize(img, (368, 368))
#     # img = transforms.ToTensor()(img)
#     #
#     # # center-map:368*368*1
#     # centermap = np.zeros((368, 368, 1), dtype=np.float32)
#     # center_map = guassian_kernel(size_h=368, size_w=368, center_x=center[0], center_y=center[1], sigma=3)
#     # center_map[center_map > 1] = 1
#     # center_map[center_map < 0.0099] = 0
#     # centermap[:, :, 0] = center_map
#     # centermap = torch.from_numpy(centermap.transpose((2, 0, 1)))
#     img, heatmap, centermap, mask = sample
#     img = torch.unsqueeze(img, 0).to(device)
#     centermap = torch.unsqueeze(centermap, 0).to(device)
#
#     model.eval()
#     # get heatmap
#     heat1, heat2, heat3, heat4, heat5, heat6 = model(img, centermap)
#     kpts = get_kpts(heat1, img_h=368.0, img_w=368.0)
#     print(kpts)
#     kpts = get_kpts(heat2, img_h=368.0, img_w=368.0)
#     print(kpts)
#     kpts = get_kpts(heat3, img_h=368.0, img_w=368.0)
#     print(kpts)
#     kpts = get_kpts(heat4, img_h=368.0, img_w=368.0)
#     print(kpts)
#     kpts = get_kpts(heat5, img_h=368.0, img_w=368.0)
#     print(kpts)
#     kpts = get_kpts(heat6, img_h=368.0, img_w=368.0)
#     print(kpts)
#     kpts0 = get_kpts(heatmap.unsqueeze(0))
#     print(kpts0)
#
#     draw_paint(img, kpts)


def test_example(model, img_path, center):
    img = np.array(cv2.resize(cv2.imread(img_path), (368, 368)), dtype=np.float32)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    # normalize
    mean = [128.0, 128.0, 128.0]
    std = [256.0, 256.0, 256.0]
    for t, m, s in zip(img, mean, std):
        t.sub_(m).div_(s)

    # center-map:368*368*1
    centermap = np.zeros((368, 368, 1), dtype=np.float32)
    center_map = guassian_kernel(size_h=368, size_w=368, center_x=center[0], center_y=center[1], sigma=3)
    center_map[center_map > 1] = 1
    center_map[center_map < 0.0099] = 0
    centermap[:, :, 0] = center_map
    centermap = torch.from_numpy(centermap.transpose((2, 0, 1)))

    img = torch.unsqueeze(img, 0)
    centermap = torch.unsqueeze(centermap, 0)

    model.eval()
    input_var = torch.autograd.Variable(img)
    center_var = torch.autograd.Variable(centermap)

    # get heatmap
    heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
    kpts = get_kpts(heat1, img_h=368.0, img_w=368.0)
    print(kpts)
    kpts = get_kpts(heat2, img_h=368.0, img_w=368.0)
    print(kpts)
    kpts = get_kpts(heat3, img_h=368.0, img_w=368.0)
    print(kpts)
    kpts = get_kpts(heat4, img_h=368.0, img_w=368.0)
    print(kpts)
    kpts = get_kpts(heat5, img_h=368.0, img_w=368.0)
    print(kpts)
    kpts = get_kpts(heat6, img_h=368.0, img_w=368.0)
    print(kpts)

    draw_paint(img_path, kpts)


def visualize(model=None):
    if not model:
        print('====== model is None ======')
        checkpoint = torch.load('BEST_checkpoint.tar')
        model = checkpoint['model']
    model.eval()
    images_path = os.listdir(path)
    images_path = [path + img_path for img_path in images_path]
    # data_set = lsp_data()
    samples = random.sample(list(images_path), 8)

    mat_arr = scipy.io.loadmat(mat_path)['joints']
    # lspnet (14,3,10000)
    kpts = mat_arr.transpose([2, 0, 1])
    for sample in samples:
        idx = int(sample.split('/')[-1][2:7])
        print(kpts[idx - 1][:, 0:2])
        test_example(model, sample, [184, 184])


idx = 0
if __name__ == '__main__':
    visualize()
