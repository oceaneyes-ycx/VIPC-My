"""ShapeNet-ViPC loader, retaining the research code's camera/scale convention.

The dataset's .dat files are pickled NumPy arrays. Only use trusted dataset files.
"""
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.splits import read_split


def rotation_x(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]).T


def rotation_y(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T


def load_points(path):
    with Path(path).open("rb") as stream:
        points = np.asarray(pickle.load(stream), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Expected a nonempty N x 3 point array: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite coordinates: {path}")
    return points


class ViPCDataset(Dataset):
    def __init__(self, root, split, category="all", training=False,
                 view_align=False, seed=1, input_points=3500):
        self.root = Path(root)
        self.records = read_split(split, category)
        self.training = training
        self.view_align = view_align
        self.seed = seed
        self.input_points = input_points
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        for folder in ("ShapeNetViPC-Partial", "ShapeNetViPC-GT", "ShapeNetViPC-View"):
            if not (self.root / folder).is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {self.root / folder}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        category, object_id, part_view = self.records[index]
        rng = random if self.training else random.Random(self.seed + index)
        image_view = part_view if self.view_align else f"{rng.randrange(24):02d}"
        relative = Path(category) / object_id
        rendering = self.root / "ShapeNetViPC-View" / relative / "rendering"
        with Image.open(rendering / f"{image_view}.png") as image:
            view = self.transform(image.convert("RGB"))
        gt = load_points(self.root / "ShapeNetViPC-GT" / relative / f"{image_view}.dat")
        partial = load_points(self.root / "ShapeNetViPC-Partial" / relative / f"{part_view}.dat")
        if len(partial) < self.input_points:
            partial = np.repeat(partial, self.input_points // len(partial) + 1, axis=0)[:self.input_points]
        if len(partial) != self.input_points:
            raise ValueError(f"Expected <= {self.input_points} partial points, got {len(partial)}: {relative}")
        if len(gt) < 2048:
            raise ValueError(f"Ground truth needs at least 2048 points: {relative}")
        metadata = np.loadtxt(rendering / "rendering_metadata.txt")
        theta_part, phi_part = np.radians(metadata[int(part_view), :2])
        theta_image, phi_image = np.radians(metadata[int(image_view), :2])
        partial = rotation_y(rotation_x(partial, -phi_part), np.pi + theta_part)
        partial = rotation_x(rotation_y(partial, np.pi - theta_image), phi_image)
        center = gt.mean(axis=0)
        gt = gt - center
        scale = np.linalg.norm(gt, axis=1).max()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Degenerate ground truth: {relative}")
        return {
            "image": view, "gt": torch.from_numpy((gt / scale).astype(np.float32)),
            "partial": torch.from_numpy(((partial - center) / scale).astype(np.float32)),
            "category": category, "key": f"{category}/{object_id}/{part_view}",
        }
