# Feature-map shifts for CNNs

This repository presents my contribution to a [KUBIG Beyond CNN team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONFERENCE/Beyond%20CNN%201%ED%8C%80%28%5E%5E%29). The team explored ways to give CNNs wider spatial context without using attention or large convolution kernels. My part focused on shifting intermediate feature maps before a ResNet-18 stage and measuring the effect on CIFAR-100 classification and the effective receptive field (ERF).

## My contribution

- Implemented deterministic channel-group shifts, per-forward random shifts, subpixel shifts with interpolation, and per-channel learnable soft shifts.
- Integrated these operators into the same CIFAR-100 ResNet-18 backbone and compared them with a baseline across seeds.
- Measured ERF from an intermediate activation to the input and presented the shift experiments at the KUBIG 25-2 conference.

In these experiments, shifts widened the measured ERF but did not improve classification accuracy. This result concerns my shift variants; the [team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONFERENCE/Beyond%20CNN%201%ED%8C%80%28%5E%5E%29) also explored other approaches.

## Code map

| File | Purpose |
| --- | --- |
| `operators.py` | Five shift operators and interpolation helpers |
| `model.py` | CIFAR ResNet-18 and operator selection |
| `config.py` | Experiment settings |
| `data.py` | CIFAR-100 transforms and loaders |
| `evaluation.py` | Accuracy, parameter count, and ERF calculation |
| `train.py` | Training, logging, checkpoints, and command-line entry point |
| `notebooks/ConvJitter.ipynb` | Cleaned original experiment notebook for provenance |

## Run an experiment

Install the packages in `requirements.txt`, then run a variant from this directory:

```bash
python train.py --variant baseline --seed 8
python train.py --variant shift_det --seed 8
python train.py --variant shift_learnable --seed 8
```

`train.py --help` lists the other variants and settings. CIFAR-100 is downloaded through `torchvision`. Checkpoints, TensorBoard logs, and datasets are generated locally and omitted here. The notebook records the original sequence of exploratory runs; the `.py` files separate those components by purpose.
