## Pretrained Models

Download prerequisite models and put them to `checkpoints` folder:

| Model | Download link | File size |
| --- | --- | --- |
| 16kHz VAE | [v1-16.pth](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth) | 655M |
| 16kHz BigVGAN vocoder (from Make-An-Audio 2) | [best_netG.pt](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt) | 429M |
| Flowley | coming soon… | — |

The (minimal) expected directory structure:

```
Flowley
├── checkpoints
|   ├── bigvgan
|   |   ├── best_netG.pt
|   ├── vae
|   |   ├── v1-16.pth
...
```