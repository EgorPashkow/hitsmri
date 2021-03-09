import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)

        x = self.c2(x)
        x = self.relu(x)

        return x

def resize_and_cat(x1, x2):
    return None

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = DoubleConv(3, 64)
        self.c2 = DoubleConv(64, 128)
        self.c3 = DoubleConv(128, 256)
        self.c4 = DoubleConv(256, 512)
        self.c5 = DoubleConv(512, 1024)

        self.c6 = DoubleConv(1024, 512)
        self.c7 = DoubleConv(512, 256)
        self.c8 = DoubleConv(256, 128)
        self.c9 = DoubleConv(128, 64)
        self.c10 = nn.Conv2d(64, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.c1(x)
        x1 = x.clone()

        x = self.pool(x)
        x = self.c2(x)
        x2 = x.clone()

        x = self.pool(x)
        x = self.c3(x)
        x3 = x.clone()

        x = self.pool(x)
        x = self.c4(x)
        x4 = x.clone()

        x = self.pool(x)
        x = self.c5(x)

        x = self.up(x)
        x = self.c6(x)
        
        x = resize_and_cat(x, x4)
        return x


if __name__ == '__main__':
    tensor = torch.rand(1, 3, 256, 256)
    net = UNet()
    net(tensor)