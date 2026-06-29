# Training Documentation

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Preparing Audio-Video-Text Features](#preparing-audio-video-text-features)
- [Training on Extracted Features](#training-on-extracted-features)

## Overview
We have put a large emphasis on making training as fast as possible.
Consequently, some pre-processing steps are required.

Namely, before starting any training, we

1. Obtain training data as videos, audios, and captions
2. (Coming soon) Generate captions capturing detailed information (with both video and sound description).
3. Encode training audios into spectrograms and then with VAE into mean/std
4. Extract CLIP features from videos
5. Extract CLIP features from text (captions)
6. Encode all extracted features into [MemoryMappedTensors](https://pytorch.org/tensordict/main/reference/generated/tensordict.MemoryMappedTensor.html) with [TensorDict](https://pytorch.org/tensordict/main/reference/tensordict.html)

## Prerequisites
1. Download the training datasets. We use [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
2. Download the corresponding VAE and vocoder model. Please see [MODELS.md](./MODELS.md)

## Preparing Audio-Video-Text Features

#### 1. TSV preparation

First, you have to prepare a tsv file containing 3 columns:
- `video_path`: MP4 absolute filepath
- `text_description`: corresponding caption of WAV file
- `split`: split of the file. Either "train", "valid", or "test".

#### 2. Change configuration file
We have already prepared a [preprocessing.yaml](../configs/preprocessing.yaml) file, containing needed parameters to extract features. Give it a look!

***TL;DR:** You need to replace the 4 filepaths `tsv_file`, `output_dir`, `tmp_dir`, and `vae_ckpt_path` according to your settings.* 

#### 3. Execute script
You can use [scripts/extract_features_from_video.py](../scripts/extract_features_from_video.py) script to extract audio, video, and text features and save them as a `TensorDict` with a TSV file containing metadata.

To run this script, use the `torchrun` utility:
```shell
CUDA_VISIBLE_DEVICES=<device> torchrun --standalone scripts/extract_features_from_video.py configs/preprocessing.yaml
```

**TIPS:**
- You can run this script with multiple GPUs (with `nproc_per_node=<n>`)
- You can change parameters of the configuration file right from the terminal by appending `--<param_name> <modified_value>` to the command. For instance,
```shell
CUDA_VISIBLE_DEVICES=<devices> torchrun --standalone --nproc_per_node=2 scripts/extract_features_from_video.py configs/preprocessing.yaml --batch_size 8
```

#### 4. Output description

Output produced in `output_dir`:

1. A directory named `{split}` containing:
    - `mean.memmap`: mean values predicted by the VAE encoder (num_videos x sequence_length x 20)
    - `std.memmap`: std values predicted by the VAE encoder (num_videos x sequence_length x 20)
    - `text_features.memmap`: text features extracted from CLIP (num_videos x 77 (sequence_length) x 1024)
    - `text_len.memmap`: actual text sequence length (num_videos)
    - `visual_features.memmap`: visual features extracted from CLIP (num_videos x 32 (4 FPS) x 768)
    - `meta.json`: metadata for the above memory mappings
2. A TSV file named `{split}.tsv` containing 2 columns:
    - `path`: video filepath
    - `caption`: corresponding captions

## Training on Extracted Features

We leverage [Accelerate](https://github.com/huggingface/accelerate) and [Hydra](https://github.com/facebookresearch/hydra/) to simplify the DDP training and evaluation.

#### 1. Prerequisites

First, you have to setup your training system by running:
```shell
accelerate config
```

If you have not set up your [WandB](https://wandb.ai/site/), do so.

We highly encourage you to thoroughly check configurations in [configs](../configs/) directory. It is crucial to ensure that every paths are correctly set.

#### 2. Training

For full training, use the following command:
```shell
accelerate launch scripts/train.py
```

**TIPS**: You can change parameters of the configuration file right from the terminal by appending `<param_name>=<param_value>` to the command. For instance,
```shell
accelerate launch scripts/train.py batch_size=16 model.single_n_heads=8
```

Any outputs from training will be stored in `logs/train/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}`.
