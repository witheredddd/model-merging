"""Uniform train/test access for the eight task experts."""
from pathlib import Path

from torchvision.datasets import ImageFolder, MNIST

ROOT = Path(__file__).resolve().parents[2]

from src.datasets.cars import PytorchStanfordCars      # noqa: E402
from src.datasets.gtsrb import PyTorchGTSRB            # noqa: E402
from src.datasets.resisc45 import RESISC45Dataset      # noqa: E402
from src.datasets.svhn import PyTorchSVHN              # noqa: E402

TASKS = ("DTD", "SUN397", "Cars", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN")


def _raw(task, split):
    d = ROOT / "datasets"
    if task == "DTD":
        return ImageFolder(d / ("dtd/train" if split == "train" else "dtd/val"))
    if task == "SUN397":
        return ImageFolder(d / ("sun397/train" if split == "train" else "sun397/val"))
    if task == "EuroSAT":
        return ImageFolder(d / "EuroSAT_splits" / split)
    if task == "Cars":
        return PytorchStanfordCars(str(d), "train" if split == "train" else "test")
    if task == "GTSRB":
        return PyTorchGTSRB(str(d), split=split)
    if task == "MNIST":
        return MNIST(str(d), train=(split == "train"))
    if task == "RESISC45":
        return RESISC45Dataset(str(d), split=split)
    if task == "SVHN":
        return PyTorchSVHN(str(d / "svhn"), split=split)
    raise ValueError(task)


def _set_transform(ds, transform):
    if transform is None:
        return ds
    attr = "transforms" if hasattr(ds, "transforms") and not hasattr(ds, "transform") else "transform"
    setattr(ds, attr, transform)
    return ds


def train_set(task, transform=None):
    return _set_transform(_raw(task, "train"), transform)


def test_set(task, transform=None):
    return _set_transform(_raw(task, "test"), transform)


def test_protocol(task, dataset):
    locations = {"DTD": "dtd/val", "SUN397": "sun397/val", "EuroSAT": "EuroSAT_splits/test"}
    return dict(task=task, n=len(dataset), split=locations.get(task, "test"))
