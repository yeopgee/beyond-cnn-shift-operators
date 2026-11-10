"""Train and compare shift variants on CIFAR-100."""
import argparse
import os
import random
import time
from typing import Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import Config
from data import get_cifar100_dataloaders
from evaluation import accuracy_top1, count_params, compute_single_cell_erf
from model import build_model

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # NOTE: deterministic can slow down; keep for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(cfg: Config) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.exp_name}/{cfg.variant}/seed{cfg.seed}/{ts}"
    run_dir = os.path.join(cfg.log_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_ckpt_path(cfg: Config) -> str:
    ckpt_dir = os.path.join(cfg.ckpt_root, cfg.exp_name, cfg.variant, f"seed{cfg.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, "best.pth")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy_top1(logits, targets) * bsz
        n += bsz

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy_top1(logits, targets) * bsz
        n += bsz

    return total_loss / n, total_acc / n

def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = cfg.device
    print("Using device:", device)

    # dirs
    run_dir = make_run_dir(cfg)
    ckpt_path = make_ckpt_path(cfg)
    writer = SummaryWriter(log_dir=run_dir)

    # data
    train_loader, test_loader = get_cifar100_dataloaders(cfg)

    # model
    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")

    # loss/opt/sched
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 90], gamma=0.2
    )

    # log fixed config
    writer.add_text("config/exp_name", cfg.exp_name)
    writer.add_text("config/variant", cfg.variant)
    writer.add_scalar("model/params", n_params, 0)

    best_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # console
        print(
            f"Epoch [{epoch:03d}/{cfg.num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"|| Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )

        # tensorboard scalars
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/test", test_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # ERF snapshots
        if epoch in cfg.erf_epochs:
            heat = compute_single_cell_erf(
                model=model,
                loader=test_loader,
                device=device,
                layer_name="layer4",
                cell_xy=(2, 2),
                channel=0,
                num_batches=cfg.erf_num_batches,
            )
            
            writer.add_image(
                f"erf/layer4_cell(2,2)_epoch{epoch}",
                heat,
                global_step=epoch,
            )

        # save best
        if cfg.save_best and test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"  -> New best: {best_acc*100:.2f}% | Saved: {ckpt_path}")

    writer.close()
    print(f"Done. Best test acc: {best_acc*100:.2f}%")
    print(f"TensorBoard log dir: {run_dir}")
    print(f"Best ckpt path: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'shift_int', 'shift_int_partial', 'shift_det', 'shift_float_05', 'shift_learnable'], default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = Config(data_dir=args.data_dir, batch_size=args.batch_size,
                 num_workers=args.num_workers, num_epochs=args.epochs,
                 variant=args.variant, seed=args.seed)
    main(cfg)
