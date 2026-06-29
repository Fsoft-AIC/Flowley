<div align="center">

<h2>Precise Video-to-Audio Generation with Cross-Modal Alignment in Latent Space</h2>

<a href="https://thanhtvt.github.io/"><strong><u>Thanh V. T. Tran</u></strong><sup>1</sup></a> &nbsp;&nbsp;&nbsp;
<a href="https://nngocson2002.github.io/"><strong><u>Ngoc-Son Nguyen</u></strong><sup>1</sup></a> &nbsp;&nbsp;&nbsp;
<a href=""><strong><u>Luong Tran</u></strong><sup>1</sup></a> &nbsp;&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/long-khanh-pham-5094b923a/"><strong><u>Long-Khanh Pham</u></strong><sup>1</sup></a> &nbsp;&nbsp;&nbsp;
<a href="https://paarthneekhara.github.io/"><strong><u>Paarth Neekhara</u></strong><sup>2</sup></a> &nbsp;&nbsp;&nbsp;
<a href="https://shehzeen.github.io/"><strong><u>Shehzeen Hussain
</u></strong><sup>2</sup></a> &nbsp;&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=rJe1704AAAAJ&hl=en"><strong><u>Van Nguyen</u></strong><sup>1</sup></a>

<sup>1</sup> FPT Software AI Center, Vietnam &nbsp;&nbsp; <sup>2</sup> NVIDIA Corporation, USA

</div>

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
BibTex is coming soon...
```

## Acknowledgement
We would like to thank the authors for their great work.
- [Make-An-Audio-2](https://github.com/bytedance/Make-An-Audio-2)
- [MMAudio](https://github.com/hkchengrex/MMAudio)
- [rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch)