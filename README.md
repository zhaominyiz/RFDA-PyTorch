# RFDA-Pytorch
Official Code for 'Recursive Fusion and Deformable Spatiotemporal Attention for Video Compression Artifact Reduction' 

ACM Multimedia 2021 (ACMMM2021) Accepted Paper 

Task: Video Quality Enhancement / Video Compression Artifact Reduction

The code will be gradually open source!


## Open Source Scheduler

1 Release RF and DSTA core code within one month after camera ready [Done]

2 Release test code and models at five QP(coming soon!)

3 Release train code (you know, in a mass )

4 Release RFDA on RGB space

5 Discuss a perceptual RF variant

Feel free to contact me if you have any problems! zhaomy20@fudan.edu.cn

## 1. Pre-request

### 1.1. Environment

- Ubuntu 20.04/18.04
- CUDA 10.1
- PyTorch 1.6
- Packages: tqdm, lmdb, pyyaml, opencv-python, scikit-image

Suppose that you have installed CUDA 10.1, then:

```bash
$ git clone --depth=1 https://github.com/zhaominyiz/RFDA-PyTorch 
$ cd RFDA-PyTorch/
$ conda create -n video python=3.7 -y
$ conda activate video
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
RFDAQP22,27,32,37,42(trained on YUV space): Coming Soon

RFDAQP37(trained on RGB space): TBD
## Results
### Comparison with State of the art Methods
![vssota](https://user-images.githubusercontent.com/43022408/128298532-eef7785f-0068-4a7f-9c74-351fe49c497c.png)

### Speed and parameter size comparison
![speedvs](https://user-images.githubusercontent.com/43022408/128298558-03a3844c-2ba2-4cc0-975e-db36c9664228.png)

## Visualization Video Demo


https://user-images.githubusercontent.com/43022408/127981531-f98ce54c-7b9d-4e12-903b-9b4bb0baf1f5.mp4

## Related Works
· Boosting the performance of video compression artifact reduction with reference frame proposals and frequency domain information [[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Xu_Boosting_the_Performance_of_Video_Compression_Artifact_Reduction_With_Reference_CVPRW_2021_paper.pdf)
· Non-local convlstm for video compression artifact reduction [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Non-Local_ConvLSTM_for_Video_Compression_Artifact_Reduction_ICCV_2019_paper.pdf) [[Code]](https://github.com/xyiyy/NL-ConvLSTM)

## Special Thanks
Our framework is based on [STDF-Pytoch](https://github.com/RyanXingQL/STDF-PyTorch). Thank [RyanXingQL](https://github.com/RyanXingQL) for his work!
