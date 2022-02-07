import glob
import random
import torch
import os.path as op
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img

def f2list(f,nf):
    f-=1########train有1个的偏移
    f2head={
        3:[0,1,2],
        4:[0,2,3],
        5:[0,3,4],
        # 6:[0,4,5],
        # 7:[0,4,6],
        # 8:[0,4,7],
        # 9:[0,4,8],
        # 10:[0,4,8,9]
    }
    # if(f<4):
    #     head=list(np.clip(list(range(f-4,f)), -1, nf - 1))#valid才会用到，因为train去除了<r的
    if(f<6):
        head=f2head[f]
    else:
        if(f%4==0):
            head = [f - 8, f - 4, f - 1]
        elif(f%4==1):
            head = [f - 9, f - 5, f - 1]
        elif(f%4==2):
            head = [f - 6, f - 2, f - 1]
        elif(f%4==3):
            head = [f - 7, f - 3, f - 1]
    if(f%4==0):
        tail = [f + 1, f + 4, f + 8]
    elif(f%4==1):
        tail = [f + 1, f + 3, f + 7]
    elif(f%4==2):
        tail = [f + 1, f + 2, f + 6]
    elif(f%4==3):
        tail = [f + 1, f + 5, f + 9]
    if(f>=nf-9):
        tail=set(tail)
        to_del=set([n for n in tail if(n>=nf)])#比nf大的删了
        tail-=to_del
        todo=sorted(list(set(list(range(f+1,f+4)))-tail))[:3-len(tail)]#使用相邻帧补充
        tail=list(tail)+todo
        tail=sorted(list(tail))
    return np.array(head+[f]+tail)+1
def f2list_valid(f,nf):
    f2head={
        3:[0,1,2],
        4:[0,2,3],
        5:[0,3,4],
    }
    if(f<3):#list(range(iter_frm - radius, iter_frm + radius + 1))
        return list(range(f-3,f+4))
    elif(f<6):
        head=f2head[f]
    else:
        if (f % 4 == 0):
            head = [f - 8, f - 4, f - 1]
        elif (f % 4 == 1):
            head = [f - 9, f - 5, f - 1]
        elif (f % 4 == 2):
            head = [f - 6, f - 2, f - 1]
        elif (f % 4 == 3):
            head = [f - 7, f - 3, f - 1]
    if (f % 4 == 0):
        tail = [f + 1, f + 4, f + 8]
    elif (f % 4 == 1):
        tail = [f + 1, f + 3, f + 7]
    elif (f % 4 == 2):
        tail = [f + 1, f + 2, f + 6]
    elif (f % 4 == 3):
        tail = [f + 1, f + 5, f + 9]
    if(f>=nf-9):
        tail=set(tail)
        to_del=set([n for n in tail if(n>=nf)])#比nf大的删了
        tail-=to_del
        todo=sorted(list(set(list(range(f+1,f+4)))-tail))[:3-len(tail)]#使用相邻帧补充
        tail=list(tail)+todo
        tail=sorted(list(tail))
    return np.array(head+[f]+tail)

class MFQEv2Dataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)


class VideoTestMFQEv2Dataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
                )
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index],
                w=self.data_info['w'][index],
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num

class MFQEv2HQDataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        black_list = []
        v_index2total_f = []
        for i in range(109):
            v_index2total_f.append(0)
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                v = int(v)
                f = int(f)
                v_index2total_f[v] += 1
        self.keys = []
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                if(v in black_list):continue
                f=int(f)
                v=int(v)
                if(f<radius+1):continue
                if(f>v_index2total_f[v]-radius):continue
                self.keys.append(tmp)
        self.v_index2total_f = v_index2total_f
        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        nfs = self.v_index2total_f[int(clip)]
        neighbor_list = f2list(int(seq), nfs)
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{str(neighbor).zfill(3)}/'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            # try:
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # except:
            #     print(seq,nfs,neighbor_list)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)

class VideoTestMFQEv2HQDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
                )
            # print("NFS=",nfs)
            for iter_frm in range(nfs):
                # lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = f2list_valid(iter_frm,nfs)
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                # print("input",iter_frm)
                # print("get",lq_indexes)
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)
            # os._exit(233)
        # print(len(self.data_info['gt_path']),"233")

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index],
                w=self.data_info['w'][index],
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num

# 一个标准的全部做进去的LMDB
class MFQEv2BetaDataset(data.Dataset):
    """MFQEv2Beta dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        self.radius = radius
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        black_list = []
        v_index2total_f = []
        for i in range(109):
            v_index2total_f.append(0)
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                v = int(v)
                f = int(f)
                v_index2total_f[v] += 1
        self.keys = []
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                if(v in black_list):continue
                f=int(f)
                v=int(v)
                if(f<radius+1):continue
                if(f>v_index2total_f[v]-radius):continue
                self.keys.append(tmp)
        self.v_index2total_f = v_index2total_f
        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        nfs = self.v_index2total_f[int(clip)]
        # neighbor_list = f2list(int(seq), nfs)
        neighbor_list = list(range(int(seq) - self.radius, int(seq) + self.radius + 1))
        neighbor_list = list(np.clip(neighbor_list, 0, nfs - 1))
        # print(seq,'to',neighbor_list)
        # os._exit(233)
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{str(neighbor).zfill(3)}/'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            # try:
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # except:
            #     print(seq,nfs,neighbor_list)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)

class MFQEv2RTDataset(data.Dataset):
    """MFQEv2RT dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([3, RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        self.radius = radius
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        black_list = []
        v_index2total_f = []
        for i in range(109):
            v_index2total_f.append(0)
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                v = int(v)
                f = int(f)
                v_index2total_f[v] += 1
        self.keys = []
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                if(v in black_list):continue
                f=int(f)
                v=int(v)
                if(f<radius+1):continue
                if(f>v_index2total_f[v]-radius):continue
                self.keys.append(tmp)
        self.v_index2total_f = v_index2total_f
        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gts = []
        for i in range(-self.radius,self.radius+1):
            img_gt_path = str(clip)+'/'+str(int(seq)+i).zfill(3)+'/'
            try:
                img_bytes = self.file_client.get(img_gt_path, 'gt')
                img_gt = _bytes2img(img_bytes)  # (H W 1)
            except:
                print("Fail to get",img_gt_path)
                print("EZ to get",key)
                os._exit(233)
            img_gts.append(img_gt)
        # get the neighboring LQ frames
        img_lqs = []
        nfs = self.v_index2total_f[int(clip)]
        # neighbor_list = f2list(int(seq), nfs)
        neighbor_list = list(range(int(seq) - self.radius, int(seq) + self.radius + 1))
        neighbor_list = list(np.clip(neighbor_list, 0, nfs - 1))
        # print(seq,'to',neighbor_list)
        # os._exit(233)
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{str(neighbor).zfill(3)}/'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            # try:
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # except:
            #     print(seq,nfs,neighbor_list)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gts, img_lqs = paired_random_crop(
            img_gts, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs = img_lqs + img_gts  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        L = len(img_results)//2
        img_lqs = torch.stack(img_results[0:-L], dim=0)
        img_gts = torch.stack(img_results[-L:], dim=0)

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gts,  # (3 [RGB] H W)
            }

    def __len__(self):
        return len(self.keys)

class VideoTestMFQEv2RTDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 RT dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2_dataset/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            'nfs': [],
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
                )
            self.data_info['w'].append(w)
            self.data_info['h'].append(h)
            self.data_info['gt_path'].append(gt_vid_path)
            self.data_info['lq_path'].append(lq_vid_path)
            self.data_info['nfs'].append(nfs)
            self.data_info['name_vid'].append(name_vid)
            self.data_info['index_vid'].append(idx_vid)

    def __getitem__(self, index):
        nfs = self.data_info['nfs'][index]
        # nfs = 100
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=nfs,
            start_frm=0,
            only_y=True
            )
        img_gts = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1 T)
        # print(img_gt.shape)
        # os._exit(233)
        # get lq frames
        img = import_yuv(
            seq_path=self.data_info['lq_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=nfs,
            start_frm=0,
            only_y=True
            )
        img_lqs = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1 T)

        # no any augmentation
        img_gts = img_gts.astype('float32')
        img_lqs = img_lqs.astype('float32')
        # to tensor
        # print(img_gts.shape,'vs_1',img_lqs.shape)
        # os._exit(233)
        img_gts = torch.from_numpy(img_gts.transpose((0,2,1,3)))
        img_lqs = torch.from_numpy(img_lqs.transpose((0,2,1,3)))
        # print(img_gts.size(),'vs_2',img_lqs.size())
        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gts,  # (T 1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return self.vid_num

    def get_vid_num(self):
        return self.vid_num