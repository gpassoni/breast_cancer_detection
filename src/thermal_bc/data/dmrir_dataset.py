"""PyTorch dataset for the public DMR-IR breast thermography database."""

import os
import re
from io import StringIO

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


class DMRIRDataset(Dataset):
    """Grayscale DMR-IR thermograms (.jpg images or .txt temperature matrices) with binary labels.

    Images are resized, min-max normalised and replicated to 3 channels so they can be fed
    to an ImageNet-style ResNet backbone.
    """

    def __init__(self, img_paths, labels, transform=None, height=244, width=244):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        if img_path.endswith(".txt"):
            img = self.load_txt_image(img_path)
            # Labels may be given as class names ("cancro" = cancer) or as 0/1 integers.
            if isinstance(label, str):
                label = 1 if label == "cancro" else 0
            label = torch.tensor(label, dtype=torch.long)
        elif img_path.endswith((".jpg", ".jpeg")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image file {img_path} not found or could not be read.")
        else:
            raise ValueError(f"Unsupported file format: {img_path}")

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = np.array(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.transform:
            img = (img * 255.0).astype(np.uint8)
            img = self.transform(img)
            img = img / 255.0
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.clip(img.astype(np.float32), 0, 1)

        img = torch.tensor(np.expand_dims(img, axis=0), dtype=torch.float32)
        img = img.expand(3, -1, -1)

        return img, label

    @staticmethod
    def load_txt_image(txt_file):
        """Load a temperature matrix exported with a decimal comma."""
        with open(txt_file) as file:
            content = file.read().replace(",", ".")
        data = np.genfromtxt(StringIO(content), delimiter=None)
        return np.array(data, dtype=np.float32)


def patient_id_from_path(path, pattern):
    """Extract the patient identifier from a DMR-IR file path using regex `pattern`.

    The first capture group is used as the ID. Raises if the pattern does not match, so a
    misconfigured pattern can never silently produce an image-level split.
    """
    match = re.search(pattern, os.path.basename(path))
    if match is None:
        raise ValueError(
            f"Could not extract a patient ID from '{os.path.basename(path)}' with "
            f"pattern {pattern!r}. Pass --patient-id-pattern matching your file names."
        )
    return match.group(1)


def grouped_split(paths, labels, groups, test_size, seed=42):
    """Split into (train, test) so that no patient appears on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(paths, labels, groups))

    def pick(seq, idx):
        return [seq[i] for i in idx]

    return (
        pick(paths, train_idx),
        pick(labels, train_idx),
        pick(groups, train_idx),
        pick(paths, test_idx),
        pick(labels, test_idx),
    )


def list_dmrir_images(database_dir):
    """Return (paths, labels); images in `abnormal/` get label 1, images in `normal/` label 0."""
    paths, labels = [], []
    for folder, label in (("abnormal", 1), ("normal", 0)):
        folder_path = os.path.join(database_dir, folder)
        files = os.listdir(folder_path)
        paths += [os.path.join(folder_path, f) for f in files]
        labels += [label] * len(files)
    return paths, labels
