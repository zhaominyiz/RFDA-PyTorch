import os
import torch,functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils
from ops.dcn.deform_conv import ModulatedDeformConv
from model.attentionlayer import DSTA
# from .attentionlayer import MultiHeadNonLocalAttention
from PIL import Image

# ==========
# Spatio-temporal deformable fusion module
# ==========
# The STDF module is implemented by RyanXingQL
# Thanks for his work! you may refer to https://github.com/RyanXingQL/STDF-PyTorch
# for more details about this.
class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # print("innc=",in_nc,",,")
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # print("OUT SIZE",out.size())
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # print("OFF_MSK",off_msk.size(),"OFF",off.size(),"MSK",msk.size())
        # print("INPUS",inputs.size())
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat

# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3,Att=None,Attname='None'):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
            )
        hid_conv_lst = []
        # if Att:
        #     print("WITH ATT")
        #     hid_conv_lst.append(MutiHeadESA(nf))
        for _ in range(nb - 2):
            hid_conv_lst += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
                ]
            if Att:
                hid_conv_lst+=[DSTA(nf)]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out

# Empty Layer
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
    
    def forward(self,x):
        return x

# ==========
# RFDA network
# ==========
class RFDA(nn.Module):
    def __init__(self,opts_dict):
        super(RFDA,self).__init__()
        self.radius = 3
        self.input_len = 2 * self.radius + 1
        self.color = opts_dict['qenet']['out_nc']
        self.ffnet = STDF(
            in_nc=opts_dict['stdf']['in_nc'] * self.input_len, 
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            deform_ks=opts_dict['stdf']['deform_ks']
            )
        # self.wpnet = SpyNet()
        # self.down = nn.Sequential(
        #     nn.Conv2d(opts_dict['stdf']['out_nc'] *2, opts_dict['stdf']['out_nc'], 3, stride=1, padding=3//2),
        #     nn.ReLU(inplace=True),
        # )
        self.fuse = nn.Sequential(
            nn.Conv2d(opts_dict['stdf']['out_nc']*2, opts_dict['stdf']['out_nc'], 3, stride=1, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(opts_dict['stdf']['out_nc'], opts_dict['stdf']['out_nc'], 3, stride=1, padding=3//2),
            nn.ReLU(inplace=True),
        )
        # in_nc=opts_dict['stdf']['in_nc'] * self.input_len * 2, 
        self.wpnet = STDF(
            in_nc=opts_dict['stdf']['out_nc'] *  2, 
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            deform_ks=1
            )
        # self.down = nn.Sequential(
        #     nn.Conv2d(opts_dict['stdf']['out_nc'], 7, 3, stride=1, padding=3//2),
        #     nn.ReLU(inplace=True),
        #     )

        # self.down2 = nn.Sequential(
        #     nn.Conv2d(opts_dict['stdf']['out_nc'], 7, 3, stride=1, padding=3//2),
        #     nn.ReLU(inplace=True),
        #     )
        self.qenetname = opts_dict['qenet']['netname']
        if opts_dict['qenet']['netname']=='default':
            att = True
            if not opts_dict['qenet'].__contains__('att') or opts_dict['qenet']['att']==False:
                att = False
            attname = 'None'
            if att:
                attname = opts_dict['qenet']['attname']

            self.qenet = PlainCNN(
                in_nc=opts_dict['qenet']['in_nc'],  
                nf=opts_dict['qenet']['nf'], 
                nb=opts_dict['qenet']['nb'], 
                out_nc=opts_dict['qenet']['out_nc'],
                Att = att,
                Attname = attname,
                )
        self.hint = None

    # x is the input reference frames
    # y is the preceding hidden state feature
    def forward(self,x,y=None):
        x = x.contiguous()
        # [B F H W]
        out = self.ffnet(x)
        if y is None:
            y = torch.zeros_like(out)
        org = out
        # [B 14 H W]
        # out = torch.cat((self.down(out),self.down2(y)),1)
        out = torch.cat((out,y),1)
        # print("1",out.size())
        # [B F H W]
        out = self.wpnet(out)
        # print(org.size(),"vs",out.size())
        out = self.fuse(torch.cat((org,out),1))*0.2 + org
        hidden = out.clone()
        return self.qenet(out) + x[:, [self.radius + i*(2*self.radius+1) for i in range(self.color)], ...],hidden
        
