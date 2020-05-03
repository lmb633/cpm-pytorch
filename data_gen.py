import torch
from torch.utils.data import Dataset
import os
import scipy.io as scio
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

path = 'data/lspet_dataset/images/'
mat_path = 'data/lspet_dataset/joints.mat'


class lsp_data(Dataset):
    def __init__(self, img_size=368, stride=8):
        self.stride = stride
        self.images_path = os.listdir(path)
        self.images_path = [path + img_path for img_path in self.images_path]
        self.images_path = sorted(self.images_path)
        self.kpts_list, self.center_list = self.read_mat_file()
        self.img_size = img_size

    def read_mat_file(self):
        # kspnet (14,3,10000)
        joints = scio.loadmat(mat_path)['joints']
        lms = joints.transpose([2, 1, 0])  # (10000,3,14)
        kpts = joints.transpose([2, 0, 1]).tolist()  # (10000,14,3)
        centers = []
        # scales = []
        for i in range(lms.shape[0]):
            image = Image.open(self.images_path[i])
            w = image.size[0]
            h = image.size[1]
            center_x = (max(lms[i][0][lms[i][0] < w]) + min(lms[i][0][lms[i][0] > 0])) / 2
            center_y = (max(lms[i][1][lms[i][1] < h]) + min(lms[i][1][lms[i][1] > 0])) / 2
            centers.append([center_x, center_y])
        return kpts, centers

    def guassian_kernel(self, size_w, size_h, center_x, center_y, sigma=3.0):
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kpts = self.kpts_list[idx]
        center = self.center_list[idx]
        height, width, _ = img.shape
        h_scale = height / self.img_size
        w_scale = width / self.img_size
        img = cv2.resize(img, (self.img_size, self.img_size))
        for i in range(len(kpts)):
            kpts[i][0] = kpts[i][0] / w_scale
            kpts[i][1] = kpts[i][1] / h_scale
        center[0] = center[0] / w_scale
        center[1] = center[1] / h_scale
        height = self.img_size
        width = self.img_size
        heatmap = np.zeros((height // self.stride, width // self.stride, len(kpts) + 1), dtype=np.float32)

        for i in range(len(kpts)):
            x = kpts[i][0] // self.stride
            y = kpts[i][1] // self.stride
            heat_map = self.guassian_kernel(size_h=height // self.stride, size_w=width // self.stride, center_x=x, center_y=y)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = self.guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1])
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map
        img = transforms.ToTensor()(img)
        heatmap = torch.tensor(heatmap)
        centermap = torch.tensor(centermap)
        return img, heatmap.permute((2, 0, 1)), centermap.permute((2, 0, 1))

    def __len__(self):
        return len(self.images_path)


if __name__ == '__main__':
    data_set = lsp_data()
    img, heatmap, centermap = data_set[86]
    print(img.shape, heatmap.shape, centermap.shape)

    # img = cv2.imread(path+'im00001.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (368, 368))
    # img = transforms.ToTensor()(img)
    # def guassian_kernel(self, size_w, size_h, center_x, center_y, sigma=3.0):
    #     gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    #     D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    #     return np.exp(-D2 / 2.0 / sigma / sigma)
    # # center-map:368*368*1
    # centermap = np.zeros((368, 368, 1), dtype=np.float32)
    # center_map = guassian_kernel(size_h=368, size_w=368, center_x=center[0], center_y=center[1], sigma=3)
    # center_map[center_map > 1] = 1
    # center_map[center_map < 0.0099] = 0
    # centermap[:, :, 0] = center_map
    # centermap = torch.from_numpy(centermap.transpose((2, 0, 1)))

    img = transforms.ToPILImage()(img)
    img.show()

    centermap = centermap * 255
    centermap = np.array(centermap.squeeze(0)).astype(np.int)
    background = Image.fromarray(centermap)
    background.show()

    # for i in range(heatmap.shape[2]):
    #     hm = heatmap[:, :, i]
    #     hm = hm * 255
    #     print(hm.shape)
    #     hm = Image.fromarray(np.array(hm).astype(np.int))
    #     hm.show()
