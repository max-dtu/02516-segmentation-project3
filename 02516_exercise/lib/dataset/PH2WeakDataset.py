import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lib.dataset.PH2Dataset import DEFAULT_PH2_PATH, PH2Dataset


def _erode_mask(mask_2d, kernel_size=11):
    """Binary erosion via max-pooling on inverted mask."""
    pad = kernel_size // 2
    inv = 1.0 - mask_2d.unsqueeze(0).unsqueeze(0)
    eroded_inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=pad)
    return (1.0 - eroded_inv).squeeze(0).squeeze(0) > 0.5


def _dilate_mask(mask_2d, kernel_size=15):
    """Binary dilation via max-pooling."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(
        mask_2d.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad
    )
    return dilated.squeeze(0).squeeze(0) > 0.5


class PH2WeakDataset(Dataset):
    """PH2 training dataset with simulated point-click weak supervision.

    Returns (X, click_target, click_mask) where:
      - click_target: (1, H, W) float tensor (1.0 at positive clicks, 0.0 elsewhere)
      - click_mask:   (1, H, W) float tensor (1.0 at clicked pixels, 0.0 elsewhere)
    """

    def __init__(
        self,
        data_path=DEFAULT_PH2_PATH,
        image_size=(256, 256),
        n_pos=10,
        n_neg=10,
        strategy="uniform",
        seed=42,
    ):
        assert strategy in ("uniform", "interior"), f"Unknown strategy: {strategy}"
        self.base = PH2Dataset(
            split="train",
            data_path=data_path,
            image_size=image_size,
            seed=seed,
        )
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.strategy = strategy
        self.seed = seed

        # Precompute deterministic clicks once per training image
        self.clicks = [self._sample_clicks(i) for i in range(len(self.base))]

    def _sample_clicks(self, idx):
        _, mask = self.base[idx]
        m = mask.squeeze(0)  # (H, W)
        rng = random.Random(self.seed * 1000 + idx)

        if self.strategy == "interior":
            pos_region = _erode_mask(m, kernel_size=11)
            neg_region = ~_dilate_mask(m, kernel_size=15)
            # Fallback to full foreground/background if lesion is too small
            if pos_region.sum().item() < self.n_pos:
                pos_region = m > 0.5
            if neg_region.sum().item() < self.n_neg:
                neg_region = m <= 0.5
        else:
            pos_region = m > 0.5
            neg_region = m <= 0.5

        pos_coords = torch.nonzero(pos_region, as_tuple=False)
        neg_coords = torch.nonzero(neg_region, as_tuple=False)

        k_pos = min(self.n_pos, len(pos_coords))
        k_neg = min(self.n_neg, len(neg_coords))

        pos_idx = rng.sample(range(len(pos_coords)), k_pos) if k_pos > 0 else []
        neg_idx = rng.sample(range(len(neg_coords)), k_neg) if k_neg > 0 else []

        click_target = torch.zeros_like(mask)
        click_mask = torch.zeros_like(mask)

        for i in pos_idx:
            r, c = pos_coords[i].tolist()
            click_target[0, r, c] = 1.0
            click_mask[0, r, c] = 1.0

        for i in neg_idx:
            r, c = neg_coords[i].tolist()
            click_target[0, r, c] = 0.0
            click_mask[0, r, c] = 1.0

        return click_target, click_mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X, _ = self.base[idx]
        click_target, click_mask = self.clicks[idx]
        return X, click_target, click_mask
