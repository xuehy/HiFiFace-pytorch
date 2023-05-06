import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False, up_sample=False, norm=True):
        super(ResBlock, self).__init__()

        main_module_list = []
        if norm:
            main_module_list += [
                nn.InstanceNorm2d(in_channel),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ]
        else:
            main_module_list += [
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ]
        if down_sample:
            main_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            main_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        if norm:
            main_module_list += [
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ]
        else:
            main_module_list += [
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            ]
        self.main_path = nn.Sequential(*main_module_list)

        side_module_list = [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)]
        if down_sample:
            side_module_list.append(nn.AvgPool2d(kernel_size=2))
        elif up_sample:
            side_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.side_path = nn.Sequential(*side_module_list)

    def forward(self, x):
        x1 = self.main_path(x)
        x2 = self.side_path(x)
        return x1 + x2


class AdaIn(nn.Module):
    def __init__(self, in_channel, vector_size):
        super(AdaIn, self).__init__()
        self.eps = 1e-5
        self.std_style_fc = nn.Linear(vector_size, in_channel)
        self.mean_style_fc = nn.Linear(vector_size, in_channel)

    def forward(self, x, style_vector):
        std_style = self.std_style_fc(style_vector)
        mean_style = self.mean_style_fc(style_vector)

        std_style = std_style.unsqueeze(-1).unsqueeze(-1)
        mean_style = mean_style.unsqueeze(-1).unsqueeze(-1)

        x = F.instance_norm(x)
        x = (1 + std_style) * x + mean_style
        return x


class AdaInResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, up_sample=False):
        super(AdaInResBlock, self).__init__()
        self.vector_size = 257 + 512
        self.up_sample = up_sample

        self.adain1 = AdaIn(in_channel, self.vector_size)
        self.adain2 = AdaIn(out_channel, self.vector_size)

        main_module_list = []
        main_module_list += [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        ]
        if up_sample:
            main_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.main_path1 = nn.Sequential(*main_module_list)

        self.main_path2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

        side_module_list = [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)]
        if up_sample:
            side_module_list.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        self.side_path = nn.Sequential(*side_module_list)

    def forward(self, x, id_vector):
        x1 = self.adain1(x, id_vector)
        x1 = self.main_path1(x1)
        x2 = self.side_path(x)

        x1 = self.adain2(x1, id_vector)
        x1 = self.main_path2(x1)

        return x1 + x2


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
    ):
        super(UpSamplingBlock, self).__init__()
        self.net = nn.Sequential(ResBlock(256, 64, up_sample=True), ResBlock(64, 16, up_sample=True), ResBlock(16, 16))
        self.i_r_net = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, 3, 3, 1, 1))
        self.m_r_net = nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        i_r = self.i_r_net(x)
        m_r = self.m_r_net(x)
        return i_r, m_r
