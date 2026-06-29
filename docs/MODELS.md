## Pretrained Models

Download prerequisite models and put them to `checkpoints` folder:

| Model    | Download link | File size |
| -------- | ------- | ------- |
| 16kHz VAE | <a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth">v1-16.pth</a> | 655M |
| 16kHz BigVGAN vocoder (from Make-An-Audio 2) |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt">best_netG.pt</a> | 429M |
| Flowley | <a href="https://drive.google.com/drive/folders/1KJlLUXE0Ddr46xx0X1-4ft9tGF9nisC7?usp=sharing">ema_final_0-60000_step=5000.pth</a> | 683M |

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