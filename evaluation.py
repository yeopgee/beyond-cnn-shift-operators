"""Accuracy, model size, and effective receptive field utilities."""
import torch
from torch import nn
from torch.utils.data import DataLoader

@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_single_cell_erf(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    layer_name: str = "layer4",
    cell_xy: tuple = (0, 0),
    channel: int = 0,
    num_batches: int = 1,
) -> torch.Tensor:
    """
    Compute single-cell ERF:
    gradient of ONE spatial cell (x,y,channel) in an intermediate layer
    w.r.t input image.
    Output: (1, H, W) normalized heatmap
    """
    model.eval()

    activations = {}

    def hook_fn(_, __, output):
        activations["feat"] = output

    # register hook
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    agg = None
    seen = 0

    for b_idx, (images, _) in enumerate(loader):
        if b_idx >= num_batches:
            break

        images = images.to(device)
        images.requires_grad_(True)

        model.zero_grad()
        _ = model(images)

        feat = activations["feat"]  # (B, C, Hf, Wf)
        y, x = cell_xy

        target = feat[:, channel, y, x].sum()
        target.backward()

        grad = images.grad.detach().abs().mean(dim=1)  # (B, H, W)

        if agg is None:
            agg = grad.sum(dim=0)
        else:
            agg += grad.sum(dim=0)

        seen += images.size(0)

    hook.remove()

    heat = agg / max(seen, 1)
    heat = heat - heat.min()
    heat = heat / heat.max()

    return heat.unsqueeze(0)  # (1, H, W)
