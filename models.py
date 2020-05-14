import torch
import torch.nn as nn
import torch.nn.functional as F


class CPM(nn.Module):
    def __init__(self, k=14):
        super(CPM, self).__init__()
        self.k = k
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        # stage1 imput origin image
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

        # stage2 imput origin image  out middle featuremap
        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # below stage are same
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.Mconv1_stage2 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.Mconv1_stage3 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.Mconv1_stage4 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.Mconv1_stage5 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.Mconv1_stage6 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

    def stage1(self, image):
        x = self.pool1_stage1(F.relu(self.conv1_stage1(image)))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x)))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x)))
        x = F.relu(self.conv4_stage1(x))
        x = F.relu(self.conv5_stage1(x))
        x = F.relu(self.conv6_stage1(x))
        x = self.conv7_stage1(x)
        return x

    def stage2_pre(self, image):
        x = self.pool1_stage2(F.relu(self.conv1_stage2(image)))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x)))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x)))
        return x

    def stage2(self, pre_stage2_map, last_stage_map, center_map):
        x = F.relu(self.conv4_stage2(pre_stage2_map))
        x = torch.cat([x, last_stage_map, center_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)
        return x

    def stage3(self, pre_stage2_map, last_stage_map, center_map):
        x = F.relu(self.conv1_stage3(pre_stage2_map))
        x = torch.cat([x, last_stage_map, center_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)
        return x

    def stage4(self, pre_stage2_map, last_stage_map, center_map):
        x = F.relu(self.conv1_stage4(pre_stage2_map))
        x = torch.cat([x, last_stage_map, center_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)
        return x

    def stage5(self, pre_stage2_map, last_stage_map, center_map):
        x = F.relu(self.conv1_stage5(pre_stage2_map))
        x = torch.cat([x, last_stage_map, center_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)
        return x

    def stage6(self, pre_stage2_map, last_stage_map, center_map):
        x = F.relu(self.conv1_stage6(pre_stage2_map))
        x = torch.cat([x, last_stage_map, center_map], dim=1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.Mconv5_stage6(x)
        return x

    def forward(self, image, center_map):
        center_map = self.pool_center(center_map)
        stage2_pre_map = self.stage2_pre(image)
        first_stage_map = self.stage1(image)
        stage2_map = self.stage2(stage2_pre_map, first_stage_map, center_map)
        stage3_map = self.stage3(stage2_pre_map, stage2_map, center_map)
        stage4_map = self.stage4(stage2_pre_map, stage3_map, center_map)
        stage5_map = self.stage5(stage2_pre_map, stage4_map, center_map)
        stage6_map = self.stage6(stage2_pre_map, stage5_map, center_map)
        return first_stage_map, stage2_map, stage3_map, stage4_map, stage5_map, stage6_map


if __name__ == '__main__':
    model = CPM()
    # data = torch.zeros((1, 3, 368, 368))
    # center_map = torch.zeros((1, 1, 368, 368))
    # out = model(data, center_map)
    # for feature in out:
    #     print(feature.shape)
    print_freq = 10
    paras = model.named_parameters()
    for i, (name, para) in enumerate(paras):
        if i % print_freq == 0:
            print(name, para.sum(), para[0].sum(), para[-1].sum())
