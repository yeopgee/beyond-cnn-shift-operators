"""Feature-map shift operators from the original experiments."""
import torch
from torch import nn
from torch.nn import functional as F

def shift_with_interpolation(
    x: torch.Tensor,
    dx: float,
    dy: float,
    padding_mode: str = "border",  # "border" = replicate edge
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Shift feature map by (dx, dy) in pixel units using bilinear interpolation.
    - dx > 0: shift RIGHT
    - dy > 0: shift DOWN
    padding_mode:
      - "border" replicates boundary (closest valid pixel), like replicate padding.
      - "zeros" does zero padding.
    """
    B, C, H, W = x.shape

    # --- Build base grid in normalized coords [-1, 1] ---
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

    # --- Convert pixel shift -> normalized shift ---
    # NOTE: with align_corners=True, step is 2/(W-1), 2/(H-1)
    denom_w = max(W - 1, 1)
    denom_h = max(H - 1, 1)
    shift_x = dx * 2.0 / denom_w
    shift_y = dy * 2.0 / denom_h

    # grid_sample samples input at "grid", so to shift output by +dx,
    # we sample input from grid - shift.
    grid[..., 0] -= shift_x
    grid[..., 1] -= shift_y

    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,  # "border" for replicate edges
        align_corners=align_corners,
    )


def shift_up(x):
    return F.pad(x, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]


def shift_down(x):
    return F.pad(x, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]


def shift_left(x):
    return F.pad(x, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]


def shift_right(x):
    return F.pad(x, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]

class IdentityOp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RandomShift4Way(nn.Module):
    """
    Random 4-way integer shift with replicate padding.

    - Each forward:
        * randomly permute channels
        * split into 4 equal groups
        * shift groups: up / down / left / right
    - Padding: replicate (NOT zero, NOT wrap)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = C // 4

        # 1️⃣ random channel permutation
        perm = torch.randperm(C, device=x.device)

        idx_up    = perm[0*g : 1*g]
        idx_down  = perm[1*g : 2*g]
        idx_left  = perm[2*g : 3*g]
        idx_right = perm[3*g : 4*g]
        idx_rest  = perm[4*g :]

        out = x.clone()

        # 2️⃣ shift UP (replicate bottom row)
        if idx_up.numel() > 0:
            tmp = x[:, idx_up]
            tmp = F.pad(tmp, (0, 0, 0, 1), mode="replicate")
            out[:, idx_up] = tmp[:, :, 1:, :]

        # 3️⃣ shift DOWN (replicate top row)
        if idx_down.numel() > 0:
            tmp = x[:, idx_down]
            tmp = F.pad(tmp, (0, 0, 1, 0), mode="replicate")
            out[:, idx_down] = tmp[:, :, :-1, :]

        # 4️⃣ shift LEFT (replicate right column)
        if idx_left.numel() > 0:
            tmp = x[:, idx_left]
            tmp = F.pad(tmp, (0, 1, 0, 0), mode="replicate")
            out[:, idx_left] = tmp[:, :, :, 1:]

        # 5️⃣ shift RIGHT (replicate left column)
        if idx_right.numel() > 0:
            tmp = x[:, idx_right]
            tmp = F.pad(tmp, (1, 0, 0, 0), mode="replicate")
            out[:, idx_right] = tmp[:, :, :, :-1]

        # 6️⃣ leftover channels: identity
        if idx_rest.numel() > 0:
            out[:, idx_rest] = x[:, idx_rest]

        return out


class DeterministicShift4Way(nn.Module):
    """
    Deterministic 4-way integer shift with replicate padding.

    - Channels are split deterministically:
        [0:g]     -> up
        [g:2g]    -> down
        [2g:3g]   -> left
        [3g:4g]   -> right
    - Padding: replicate
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = C // 4

        idx_up    = slice(0*g, 1*g)
        idx_down  = slice(1*g, 2*g)
        idx_left  = slice(2*g, 3*g)
        idx_right = slice(3*g, 4*g)
        idx_rest  = slice(4*g, C)

        out = x.clone()

        # shift UP
        if g > 0:
            tmp = x[:, idx_up]
            tmp = F.pad(tmp, (0, 0, 0, 1), mode="replicate")
            out[:, idx_up] = tmp[:, :, 1:, :]

        # shift DOWN
        if g > 0:
            tmp = x[:, idx_down]
            tmp = F.pad(tmp, (0, 0, 1, 0), mode="replicate")
            out[:, idx_down] = tmp[:, :, :-1, :]

        # shift LEFT
        if g > 0:
            tmp = x[:, idx_left]
            tmp = F.pad(tmp, (0, 1, 0, 0), mode="replicate")
            out[:, idx_left] = tmp[:, :, :, 1:]

        # shift RIGHT
        if g > 0:
            tmp = x[:, idx_right]
            tmp = F.pad(tmp, (1, 0, 0, 0), mode="replicate")
            out[:, idx_right] = tmp[:, :, :, :-1]

        # leftover channels: identity
        if 4 * g < C:
            out[:, idx_rest] = x[:, idx_rest]

        return out


class RandomShift4WayPartial(nn.Module):
    """
    Random 4-way integer shift with fixed ratio + identity channels.

    Example (C=64, n_shift=10):
        - 10 up
        - 10 down
        - 10 left
        - 10 right
        - 24 identity

    Each forward:
        * randomly permute channels
        * apply deterministic split sizes
    """

    def __init__(self, n_shift: int = 10):
        super().__init__()
        self.n_shift = n_shift  # per direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.n_shift

        assert 4 * k <= C, "Not enough channels for requested shift split."

        # 1️⃣ random channel permutation
        perm = torch.randperm(C, device=x.device)

        idx_up    = perm[0*k : 1*k]
        idx_down  = perm[1*k : 2*k]
        idx_left  = perm[2*k : 3*k]
        idx_right = perm[3*k : 4*k]
        idx_rest  = perm[4*k :]

        out = x.clone()

        # 2️⃣ shift UP (replicate bottom row)
        if idx_up.numel() > 0:
            tmp = F.pad(x[:, idx_up], (0, 0, 0, 1), mode="replicate")
            out[:, idx_up] = tmp[:, :, 1:, :]

        # 3️⃣ shift DOWN (replicate top row)
        if idx_down.numel() > 0:
            tmp = F.pad(x[:, idx_down], (0, 0, 1, 0), mode="replicate")
            out[:, idx_down] = tmp[:, :, :-1, :]

        # 4️⃣ shift LEFT (replicate right column)
        if idx_left.numel() > 0:
            tmp = F.pad(x[:, idx_left], (0, 1, 0, 0), mode="replicate")
            out[:, idx_left] = tmp[:, :, :, 1:]

        # 5️⃣ shift RIGHT (replicate left column)
        if idx_right.numel() > 0:
            tmp = F.pad(x[:, idx_right], (1, 0, 0, 0), mode="replicate")
            out[:, idx_right] = tmp[:, :, :, :-1]

        # 6️⃣ identity channels
        if idx_rest.numel() > 0:
            out[:, idx_rest] = x[:, idx_rest]

        return out


class RandomShift4WayFloat(nn.Module):
    """
    Random 4-way *float* shift (bilinear interpolation).

    Example (C=64):
        - 16 up
        - 16 down
        - 16 left
        - 16 right

    shift_px: float (e.g., 0.5)
    padding_mode="border" replicates boundary values.
    """

    def __init__(self, shift_px: float = 0.5):
        super().__init__()
        self.shift_px = float(shift_px)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C % 4 == 0, "Channel dimension must be divisible by 4."

        g = C // 4
        perm = torch.randperm(C, device=x.device)

        idx_up    = perm[0*g : 1*g]
        idx_down  = perm[1*g : 2*g]
        idx_left  = perm[2*g : 3*g]
        idx_right = perm[3*g : 4*g]

        out = x.clone()
        s = self.shift_px

        out[:, idx_up]    = shift_with_interpolation(x[:, idx_up],    dx=0.0, dy=-s)
        out[:, idx_down]  = shift_with_interpolation(x[:, idx_down],  dx=0.0, dy=+s)
        out[:, idx_left]  = shift_with_interpolation(x[:, idx_left],  dx=-s,  dy=0.0)
        out[:, idx_right] = shift_with_interpolation(x[:, idx_right], dx=+s,  dy=0.0)

        return out


class LearnableShift4Way(nn.Module):
    """
    Channel-wise learnable soft shift (NO conv, NO interpolation)

    y_c = sum_d softmax(logits[c])_d * S_d(x_c)

    directions:
      [up, down, left, right, identity]
    """

    def __init__(
        self,
        num_channels: int = 64,
        temperature: float = 1.0,
        init_identity_bias: float = 0.3,   # 🔹 거의 0에 가까운 bias
        init_noise_std: float = 1e-3,      # 🔹 symmetry breaking
    ):
        super().__init__()
        self.num_channels = num_channels
        self.temperature = temperature

        # logits: (C, 5)
        self.logits = nn.Parameter(torch.zeros(num_channels, 5))

        # small identity bias
        self.logits.data[:, 4] = init_identity_bias

        # tiny noise to avoid symmetric collapse
        self.logits.data += init_noise_std * torch.randn_like(self.logits.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.num_channels, "Channel mismatch"

        # soft direction weights
        probs = F.softmax(self.logits / self.temperature, dim=-1)  # (C,5)

        # shifted versions
        x_up    = shift_up(x)
        x_down  = shift_down(x)
        x_left  = shift_left(x)
        x_right = shift_right(x)
        x_id    = x

        stacked = torch.stack(
            [x_up, x_down, x_left, x_right, x_id],
            dim=2,   # (B, C, 5, H, W)
        )

        probs = probs.view(1, C, 5, 1, 1)
        out = (stacked * probs).sum(dim=2)

        return out
