## Pretrained Models

Download prerequisite models and put them to `checkpoints` folder:

| Model | Download link | File size |
| --- | --- | --- |
| 16kHz VAE | [v1-16.pth](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth) | 655M |
| 16kHz BigVGAN vocoder (from Make-An-Audio 2) | [best_netG.pt](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt) | 429M |
| Flowley | [ema_final_0-60000_step=5000.pth](https://huggingface.co/Fsoft-AIC/Flowley/resolve/main/ema_final_0-60000_step%3D5000.pth) | 685M |

The (minimal) expected directory structure:

```
Flowley
├── checkpoints
|   ├── bigvgan
|   |   ├── best_netG.pt
|   ├── vae
|   |   ├── v1-16.pth
...
├── logs
|   ├── train
|   |   |   ├── runs
|   |   |   |   ├── pretrained_ckpt
|   |   |   |   |   ├── ema_final_0-60000_step=5000.pth
...
```