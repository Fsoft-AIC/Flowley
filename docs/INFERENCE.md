# Inference Documentation

## Table of Contents
- [Preparing Audio-Video-Text Features](#preparing-audio-video-text-features)
- [Sampling on Extracted Features](#sampling-on-extracted-features)

## Preparing Audio-Video-Text Features

Download the benchmark dataset (table below) and follow instruction detailed in [TRAINING.md](./TRAINING.md).

| Dataset    | Download link |
| -------- | ------- |
| VGGSound | <a href="https://www.robots.ox.ac.uk/~vgg/data/vggsound">vggsound</a> |
| Movie Gen Audio Bench |<a href="https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenAudioBenchSfx.tar.gz">MovieGenAudioBenchSfx</a> |

## Sampling on Extracted Features

After training the model, you will find the logged checkpoints, configuration files, and other outputs in the log directory `<log_dir>`. To sample using the pretrained checkpoints, run:
```shell
accelerate launch scripts/sample.py run_dir=<log_dir>
```

To sample using our checkpoints, execute:
```shell
accelerate launch scripts/sample.py --config_file configs/single_gpu_accelerate_config.yaml --gpu_ids='0' scripts/sample.py run_dir=./logs/train/runs/pretrained_ckpt max_step_ema=60000 cfg_strength=7.5
```

**TIPS**: You can change parameters of the configuration file right from the terminal by appending `<param_name>=<param_value>` to the command. For instance,
```shell
accelerate launch scripts/sample.py cfg_strength=4.0
```

For other parameters, refer to the [sample.yaml](../configs/sample.yaml) file.