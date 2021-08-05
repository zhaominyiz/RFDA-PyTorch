# RFDA-Pytorch
Official Code for 'Recursive Fusion and Deformable Spatiotemporal Attention for Video Compression Artifact Reduction' 

ACM Multimedia 2021 (ACMMM2021) Accepted Paper 

Task: Video Quality Enhancement / Video Compression Artifact Reduction

The code will be gradually open source!


## Open Source Scheduler

1 Release RF and DSTA core code within one month after camera ready [Done]

2 Release train code (you know, in a mass ) [TBD]

Feel free to contact me if you have any problems! zhaomy20@fudan.edu.cn

## 1. Pre-request

### 1.1. Environment

- Ubuntu 20.04/18.04
- CUDA 10.1
- PyTorch 1.6
- Packages: tqdm, lmdb, pyyaml, opencv-python, scikit-image

Suppose that you have installed CUDA 10.1, then:

```bash
$ git clone --depth=1 https://github.com/RyanXingQL/STDF-PyTorch 
$ cd STDF-PyTorch/
$ conda create -n stdf python=3.7 -y
$ conda activate stdf
$ python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
$ python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

### 1.2. DCNv2

**Build DCNv2.**

```bash
$ cd ops/dcn/
$ bash build.sh
```

**(Optional) Check if DCNv2 works.**

```bash
$ python simple_check.py
```

> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

## Train

## Test
### Test MFQE 2.0 dataset
### Test your own video clip
## Pretrain models
RFDAQP22,27,32,37,42: TBD
## Visualization Video Demo


https://user-images.githubusercontent.com/43022408/127981531-f98ce54c-7b9d-4e12-903b-9b4bb0baf1f5.mp4


## Special Thanks
Our framework is based on STDF-Pytoch. Thank RyanXingQL for his work!
