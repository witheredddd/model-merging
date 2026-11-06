import os
import torch
from torchvision.datasets import SVHN as OriginalSVHN
from pathlib import Path
import scipy.io

class PyTorchSVHN(OriginalSVHN):
    def download(self):
        if self._check_integrity():
            return
        return super().download()

    def _check_integrity(self):
        marker = Path(self.root) / "mirror_verified.json"
        if marker.exists():
            p = Path(self.root) / self.filename
            if p.exists():
                shapes = {name: shape for name, shape, _ in scipy.io.whosmat(p)}
                expected = 73257 if self.split == "train" else 26032
                return shapes.get("X") == (32, 32, 3, expected) and shapes.get("y") == (expected, 1)
        return super()._check_integrity()
import numpy as np


class SVHN:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):
        # to fit with repo conventions for location
        modified_location = os.path.join(location, "svhn")

        self.train_dataset = PyTorchSVHN(
            root=modified_location, download=True, split="train", transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location, download=True, split="test", transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
