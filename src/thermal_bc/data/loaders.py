"""DataLoader factory shared by the segmentation scripts."""

from torch.utils.data import DataLoader


def make_loader(dataset, batch_size, train):
    """Loader with the settings used for all segmentation experiments."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=train,
    )
