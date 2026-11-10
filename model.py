"""CIFAR ResNet-18 with a configurable shift before stage 1."""
from typing import Optional
import torch
from torch import nn
from config import Config
from operators import (IdentityOp, RandomShift4Way, DeterministicShift4Way,
                       RandomShift4WayPartial, RandomShift4WayFloat, LearnableShift4Way)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNetCIFAR18(nn.Module):
    """
    CIFAR ResNet-18:
      Stem: 3x3 conv, stride=1 (no 7x7, no maxpool)
      Stages: [2,2,2,2] BasicBlocks
    Operator plug-in:
      - stage1_pre_op is applied right BEFORE Stage1
      - stage1_post_op is applied right AFTER Stage1
    """
    def __init__(
        self,
        num_classes: int = 100,
        stage1_pre_op: Optional[nn.Module] = None,
        stage1_post_op: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_ch = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Operators (default Identity)
        self.stage1_pre_op = stage1_pre_op if stage1_pre_op is not None else IdentityOp()
        self.stage1_post_op = stage1_post_op if stage1_post_op is not None else IdentityOp()

        # Stages
        self.layer1 = self._make_layer(64,  2, stride=1)  # Stage1: 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # Stage2: 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # Stage3:  8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # Stage4:  4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_ch, out_ch, stride=s))
            self.in_ch = out_ch
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))   # (B,64,32,32)

        # Stage1 (intervention point)
        x = self.stage1_pre_op(x)
        x = self.layer1(x)                       # (B,64,32,32)
        x = self.stage1_post_op(x)

        # Rest
        x = self.layer2(x)                       # (B,128,16,16)
        x = self.layer3(x)                       # (B,256, 8, 8)
        x = self.layer4(x)                       # (B,512, 4, 4)

        x = self.avgpool(x)                      # (B,512,1,1)
        x = torch.flatten(x, 1)
        x = self.fc(x)                           # (B,100)
        return x


def build_model(cfg: Config) -> nn.Module:
    """
    Build model according to cfg.variant.
    ONLY this function controls operator choice.
    """

    if cfg.variant == "baseline":
        pre_op  = IdentityOp()
        post_op = IdentityOp()

    elif cfg.variant == "shift_int":
        pre_op  = RandomShift4Way()
        post_op = IdentityOp()

    elif cfg.variant == "shift_int_partial":
        pre_op  = RandomShift4WayPartial(n_shift=10)
        post_op = IdentityOp()

    elif cfg.variant == "shift_det":
        pre_op  = DeterministicShift4Way()
        post_op = IdentityOp()

    elif cfg.variant == "shift_float_05":
        pre_op  = RandomShift4WayFloat(shift_px=0.5)
        post_op = IdentityOp()

    elif cfg.variant == "shift_learnable":
        pre_op = LearnableShift4Way(
            num_channels=64,
            temperature=1.0,
            init_identity_bias=0.1,
        )
        post_op = IdentityOp()

    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")

    return ResNetCIFAR18(
        num_classes=cfg.num_classes,
        stage1_pre_op=pre_op,
        stage1_post_op=post_op,
    )
