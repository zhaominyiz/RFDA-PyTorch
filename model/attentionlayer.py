import math
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
from torch.nn import functional as F
from math import floor, ceil
from ops.dcn.deform_conv import ModulatedDeformConv

def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
       if not padding and stride==1:
           padding = kernel_size // 2
       return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)

class DSTA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(DSTA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        # DCN is better
        self.dcn = ModulatedDeformConv(f,f,3,padding=1,deformable_groups=f)
        self.mask = conv(f,f*3*3*3,3,padding=1)
        # two mask, multilevel fusion
        self.f = f
        self.down_conv2 = nn.Sequential(
                nn.Conv2d(f, f, 3, stride=2, padding=3//2),
                nn.ReLU(inplace=True))
        self.mask2 = conv(f,f*3*3*3,3,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(f, 2*f, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*f, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        f = x.clone()
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = self.relu(c3)
        dc3 = self.down_conv2(c3)
        off_mask2 = self.mask2(dc3)
        off_msk = self.mask(c3)
        off_mask2 = F.interpolate(off_mask2, (off_msk.size(2), off_msk.size(3)), mode='bilinear', align_corners=False)
        off_msk = off_msk + off_mask2
        off = off_msk[:, :self.f*2*3*3, ...]
        msk = torch.sigmoid(
            off_msk[:, self.f*2*3*3:, ...]
            )
        c3 = self.dcn(v_max,off,msk)
        c3 = F.relu(c3,inplace = True)
        y = self.avg_pool(c3)
        y = self.conv_du(y)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        # print(x.size(),'vs',m.size(),'vs',y.size())
        # is you wanna visualize them
        # map_m = transforms.ToPILImage()(m[0,0,...]).convert('L')
        # map_m.save("./map_m1.png") # 
        # map_m = transforms.ToPILImage()(m[0,2,...]).convert('L')
        # map_m.save("./map_m2.png") # 
        # map_m = transforms.ToPILImage()(m[0,4,...]).convert('L')
        # map_m.save("./map_m3.png") # 
        # map_m = transforms.ToPILImage()(m[0,6,...]).convert('RGB')
        # map_m.save("./map_m4.png") # 
        # map_m = transforms.ToPILImage()(m[0,8,...]).convert('RGB')
        # map_m.save("./map_m5.png") # 
        # print("y=",y[0])
        return x * m * y
