"""Download CIFAR-100 and construct training and test loaders."""
from typing import Tuple
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import Config

def get_cifar100_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=cfg.data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=cfg.data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
