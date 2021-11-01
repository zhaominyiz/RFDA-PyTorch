# :sparkles: RFDA-Pytorch :sparkles:
Official Code for 'Recursive Fusion and Deformable Spatiotemporal Attention for Video Compression Artifact Reduction' 

ACM Multimedia 2021 (ACMMM2021) Accepted Paper 

Task: Video Quality Enhancement / Video Compression Artifact Reduction

The code will be gradually open source!


## Open Source Scheduler

1 Release RF and DSTA core code within one month after camera ready [Done]

2 Release test code and models at five QP [Done]

3 Release train code (you know, in a mass ) [TBD]

4 Release RFDA on RGB space

5 Discuss a perceptual RF variant

## :e-mail: Contact :e-mail:
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

## :fire: 2. Train :fire:

## :zap: 3. Test :zap:
### 3.1 Test MFQE 2.0 dataset
Please build the MFQE 2.0 dataset first (See [Here](https://github.com/RyanXingQL/STDF-PyTorch)), then run test_yuv_RF.py.

More instructions will coming soon!
```bash
$ python test_yuv_RF.py --opt_path cofig/****.yml
```
### 3.2 Test your own video clip
For yuv videos, you may refer to test_one_video_yuv_RF.py.
```bash
$ python test_one_video_yuv_RF.py --opt_path cofig/****.yml
```

For rgb videos, we will update new model and codes soon.
## :seedling: 3.3 Pretrain models :seedling:
RFDAQP22,27,32,37,42(trained on YUV space): [BaiduDisk](https://pan.baidu.com/s/1Py4_2-I5gq9LuoudKLZoUA) (RFDA) [GoogleDisk](https://drive.google.com/file/d/1HbNgmr4sxAxa4jaek7WLbqB4gHOhjKn0/view?usp=sharing)

RFDAQP37(trained on RGB space): ToBeDone!
## :beers: Results :beers:
### Comparison with State of the Art Methods
![vssota](https://user-images.githubusercontent.com/43022408/128298532-eef7785f-0068-4a7f-9c74-351fe49c497c.png)

### Speed and parameter size comparison
![speedvs](https://user-images.githubusercontent.com/43022408/128298558-03a3844c-2ba2-4cc0-975e-db36c9664228.png)

## :sparkling_heart: Visualization Video Demo :sparkling_heart:


https://user-images.githubusercontent.com/43022408/127981531-f98ce54c-7b9d-4e12-903b-9b4bb0baf1f5.mp4

## :wink: Related Works :wink:
· Boosting the performance of video compression artifact reduction with reference frame proposals and frequency domain information [[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Xu_Boosting_the_Performance_of_Video_Compression_Artifact_Reduction_With_Reference_CVPRW_2021_paper.pdf)

· Non-local convlstm for video compression artifact reduction [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Non-Local_ConvLSTM_for_Video_Compression_Artifact_Reduction_ICCV_2019_paper.pdf) [[Code]](https://github.com/xyiyy/NL-ConvLSTM)

## :satisfied: Citation :satisfied:
If you find this project is useful for your research, please cite:
```
@article{zhao2021recursive,
  title={Recursive Fusion and Deformable Spatiotemporal Attention for Video Compression Artifact Reduction},
  author={Zhao, Minyi and Xu, Yi and Zhou, Shuigeng},
  journal={arXiv preprint arXiv:2108.02110},
  year={2021}
}
```

## :thumbsup: Special Thanks :thumbsup:
Our framework is based on [STDF-Pytoch](https://github.com/RyanXingQL/STDF-PyTorch). Thank [RyanXingQL](https://github.com/RyanXingQL) for his work!
