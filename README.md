## Precise Video-to-Audio Generation with Cross-Modal Alignment in Latent Space

[**<u>Thanh V. T. Tran</u**>1](https://thanhtvt.github.io/)     [**<u>Ngoc-Son Nguyen</u**>1](https://nngocson2002.github.io/)     [**<u>Luong Tran</u**>1](https://khanhluong34.github.io/)     [**<u>Long-Khanh Pham</u**>1](https://www.linkedin.com/in/long-khanh-pham-5094b923a/)     [**<u>Paarth Neekhara</u**>2](https://paarthneekhara.github.io/)     [**<u>Shehzeen Hussain</u>** 2](https://shehzeen.github.io/)     [**<u>Van Nguyen</u**>1](https://scholar.google.com/citations?user=rJe1704AAAAJ&hl=en)

1 FPT Software AI Center, Vietnam    2 NVIDIA Corporation, USA

## Table of Contents

- [Introduciton](#introduction)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Introduction

**Flowley** is single-stage architecture that synthesizes temporally and semantically aligned soundtracks from silent videos. The key architecture innovation is **Progressive Soft-masked Cross-Attention (PSCA)**, a novel mechanism that embeds precise temporal prior directly within the attention layers.

We also propose **SoundCap**, a plug-and-play pipeline leveraging audio-visual LLMs to generate detailed, sound-oriented captions that robustly guide our model.

Checkout our [project page](https://flowley-v2a.github.io/) for more generated samples.

## Installation

We have only tested this on Ubuntu 22.04 and PyTorch 2.0+

#### 1. Install dependencies

We use PyTorch 2.2 with CUDA 11.8. Please check if your GPUs/driver support this.

```shell
cd Flowley
conda env create -f environment.yml
```

#### 2. Install project

```shell
pip install -e .
```

## Training

See [TRAINING.md](./docs/TRAINING.md)

## Inference

See [INFERENCE.md](./docs/INFERENCE.md)

## Citation

If our paper or codebase aids your research, please consider citing us:

```
@article{tran2026precise,
  author  = {Thanh V. T. Tran and Ngoc-Son Nguyen and Luong Tran and Long-Khanh Pham and Paarth Neekhara and Shezheen Hussain and Van Nguyen},
  title   = {Precise Video-to-Audio Generation with Cross-Modal Alignment in Latent Space},
  journal = {arXiv preprint arXiv:2607.06405},
  year    = {2026},
  url     = {https://arxiv.org/pdf/2607.06405}
}
```

## Acknowledgement

We would like to thank the authors for their great work.

- [Make-An-Audio-2](https://github.com/bytedance/Make-An-Audio-2)
- [MMAudio](https://github.com/hkchengrex/MMAudio)
- [rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch)