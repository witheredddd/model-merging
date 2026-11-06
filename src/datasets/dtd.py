import os
import shutil
from pathlib import Path

import torch
import torchvision.datasets as datasets


def _materialize_split(image_root: Path, split_file: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for relative_path in split_file.read_text(encoding="utf-8").splitlines():
        if not relative_path:
            continue
        source = image_root / relative_path
        class_name = Path(relative_path).parts[0]
        destination = output_root / class_name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


def prepare_dtd_dataset(location=os.path.expanduser("~/data")) -> Path:
    dataset_root = Path(location) / "dtd"
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"

    if train_root.exists() and val_root.exists():
        return dataset_root

    # Download the official torchvision dataset, then materialize the
    # train/val directory layout expected by this repository's ImageFolder loader.
    datasets.DTD(root=location, split="train", partition=1, download=True)
    datasets.DTD(root=location, split="val", partition=1, download=True)

    source_root = dataset_root / "dtd" if (dataset_root / "dtd").exists() else dataset_root
    image_root = source_root / "images"
    labels_root = source_root / "labels"
    _materialize_split(image_root, labels_root / "train1.txt", train_root)
    _materialize_split(image_root, labels_root / "val1.txt", val_root)
    return dataset_root


class DTD:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=6,
    ):
        dataset_root = prepare_dtd_dataset(location)
        traindir = dataset_root / "train"
        valdir = dataset_root / "val"

        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [
            idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))
        ]
