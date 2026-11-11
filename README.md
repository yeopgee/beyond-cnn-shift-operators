# Expanding CNN receptive fields with feature-map shifts

This repository contains my shift-operation experiments from the [KUBIG Beyond CNN team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONFERENCE/Beyond%20CNN%201%ED%8C%80%28%5E%5E%29). For more information about the project, see the team repository.

## Motivation

- Shifting feature map could widen the effective receptive field without relying on larger convolution kernels or attention, while adding little computation.
- Random shifts could also act as feature-map-level augmentation and encourage more robust representations.

![Illustration of jittering a feature map](assets/jittering-concept.png)

*The motivating idea from my KUBIG presentation: shift features before local convolution.*

## My contribution

- Implemented **deterministic shifts**, **random shifts**, **fractional shifts**, and **learnable shifts**.
- Inserted each operator into the same ResNet-18 backbone and held the remaining architecture and training setup fixed for comparison.
- Compared classification accuracy and visualized the **effective receptive field (ERF)** at a fixed output position.

![Diagram of feature-map shifts before a CNN layer](assets/shift-diagram.png)

*Shift-operation diagram from my KUBIG presentation.*

## Results

CIFAR-100 test accuracy after 120 epochs across four seeds:

| Operator | Mean accuracy (%) | SD |
| --- | ---: | ---: |
| No shift | **77.62** | 0.16 |
| Deterministic shift | 77.18 | 0.24 |
| Learnable shift | 77.01 | 0.21 |
| fractional shift | 76.65 | 0.31 |
| Random shift | 76.17 | 0.21 |

At epoch 20, the shifted models showed a wider measured ERF:

| No shift | Deterministic shift | Learnable shift |
| :---: | :---: | :---: |
| <img src="assets/erf-baseline.png" alt="No-shift ERF" width="240" height="188"> | <img src="assets/erf-deterministic.png" alt="Deterministic-shift ERF" width="240" height="188"> | <img src="assets/erf-learnable.png" alt="Learnable-shift ERF" width="240" height="188"> |

*ERF visualizations from my KUBIG presentation, shown at the same training epoch.*

The shifts widened the measured ERF but did not improve classification accuracy in these experiments. The [team project](https://github.com/KU-BIG/KUBIG_2025_FALL/tree/main/KUBIG%20CONFERENCE/Beyond%20CNN%201%ED%8C%80%28%5E%5E%29) also explored other approaches, including incorporating fully connected layers and rotations.

## Interpretations

- My shift operations act as channel-wise convolutions with randomly selected, fixed, or constrained learnable kernels.
- Although the deterministic and learnable shifts preserved translation equivariance away from boundaries, their restrictions on the shift kernels introduced additional inductive biases whose benefits for the task remained unclear.
- Random shifts may destabilize training, because shift directions are selected randomly at each forward pass.
 
## Code structure

| File | Role |
| --- | --- |
| [`operators.py`](operators.py) | Shift operators and interpolation |
| [`model.py`](model.py) | ResNet-18 with operator selection |
| [`train.py`](train.py) | Training and experiment entry point |
| [`evaluation.py`](evaluation.py) | Accuracy and ERF measurement |
| [`notebooks/ConvJitter.ipynb`](notebooks/ConvJitter.ipynb) | Original experiment sequence |
