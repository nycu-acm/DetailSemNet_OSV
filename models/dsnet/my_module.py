import torch
import torch.nn as nn
import torch.nn.functional as F

class HighMixer_v2(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        **kwargs, ):
        super().__init__()
        
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.mid_gelu = nn.GELU()

    def forward(self, x):
        # B, C H, W
        
        cx = self.conv(x)
        cx = self.mid_gelu(cx)
        
        return cx


class HighMixer_v3(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        **kwargs, ):
        super().__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1, padding=0)
        self.mid_gelu = nn.GELU()

    def forward(self, x):
        # B, C H, W
        
        x = self.Maxpool(x)
        cx = self.conv(x)
        cx = self.mid_gelu(cx)
        
        return cx