import torch.nn as nn
import torch
import math


def pixel_shuffle_3d(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x):
        tx = x
        x = self.prelu(self.conv1(x))
        x = self.conv2(x) + tx
        return x


class SubPixel(nn.Module):
    def __init__(self, channels, upscale_factor):
        super(SubPixel, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(channels, channels * (upscale_factor ** 3), 3, stride=1, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(pixel_shuffle_3d(self.conv(x), self.upscale_factor))
        return x

def primes(n):
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac


class SRCNN(nn.Module):
    def __init__(self, channels, block_num, post_block_num, upscale, in_channels):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, 9, stride=1, padding=4)
        self.prelu1 = nn.PReLU()
        self.residual_blocks = nn.ModuleList([ResBlock(channels) for k in range(block_num)])
        self.conv2 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.sub_pixel_blocks = nn.ModuleList([SubPixel(channels, k) for k in primes(upscale)])
        self.post_residual_blocks = nn.ModuleList([ResBlock(channels) for k in range(post_block_num)])
        self.conv3 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(channels, in_channels, 5, stride=1, padding=2)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))  # B * 64 * L * L * L

        if len(self.residual_blocks) != 0:
            tx = x
            for residual_block in self.residual_blocks:
                x = residual_block(x)
            x = self.conv2(x) + tx

        for sub_pixel_block in self.sub_pixel_blocks:
            x = sub_pixel_block(x)

        if len(self.post_residual_blocks) != 0:
            tx = x
            for residual_block in self.post_residual_blocks:
                x = residual_block(x)
            x = self.conv3(x) + tx

        x = self.conv4(x)
        return x


class Model(nn.Module):
    def __init__(self, channels, block_num, post_block_num, upscale):
        super(Model, self).__init__()
        self.srcnn1 = SRCNN(channels, block_num, post_block_num, upscale, 1)
        self.srcnn2 = SRCNN(channels, block_num, post_block_num, upscale, 1)
        self.srcnn3 = SRCNN(channels, block_num, post_block_num, upscale, 1)

    def forward(self, t):
        x = self.srcnn1(t[:, 0: 1])
        y = self.srcnn2(t[:, 1: 2])
        z = self.srcnn3(t[:, 2: 3])
        return torch.cat([x, y, z], dim=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


def down_4(x):
    for i in range(4):
        if x % 2 == 0:
            x = x //2
        else:
            x = (x + 1) // 2
    return x


class Discriminator(nn.Module):
    def __init__(self, x, y, z):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, 3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, 3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, 128, 3, stride=2, padding=1)
        self.conv7 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.conv8 = nn.Conv3d(256, 256, 3, stride=2, padding=1)
        t = down_4(x) * down_4(y) * down_4(z)
        self.linear1 = nn.Linear(t * 256, 1024)
        #self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x)) # B * 3 * x * y * z -> B * 64 * x * y * z
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x)) # B * 512 * x/16 * y/16 * z/16
        x = self.relu(self.linear1(x.reshape(x.shape[0], -1)))
        #x = self.relu(self.linear1(torch.mean(x.reshape(x.shape[0], 256, -1), dim=2)))
        return torch.sigmoid(self.linear2(x).reshape(-1))
