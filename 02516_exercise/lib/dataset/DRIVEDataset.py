import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

DEFAULT_DRIVE_PATH = "/dtu/datasets1/02516/DRIVE"


class DRIVEDataset(Dataset):
    """PyTorch Dataset for the DRIVE retinal blood vessel segmentation dataset.

    Note: Only DRIVE/training/ contains manual vessel ground-truth masks (1st_manual/).
    DRIVE/test/ only contains field-of-view masks (mask/), not vessel masks.
    Therefore, we split the 20 labeled images in DRIVE/training/ into train/val/test.
    """

    def __init__(
        self,
        split="train",
        data_path=DEFAULT_DRIVE_PATH,
        image_size=(512, 512),
        split_ratios=(0.7, 0.15, 0.15),
        seed=42,
        joint_transform=None,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"

        self.split = split
        self.data_path = data_path
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.joint_transform = joint_transform

        train_dir = os.path.join(data_path, "training")
        img_dir = os.path.join(train_dir, "images")
        mask_dir = os.path.join(train_dir, "1st_manual")

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(
                f"Expected DRIVE training/images and training/1st_manual under {data_path}"
            )

        all_pairs = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".tif"):
                continue
            # e.g. '21_training.tif' -> '21_manual1.gif'
            case_id = fname.split("_")[0]
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, f"{case_id}_manual1.gif")
            if os.path.isfile(mask_path):
                all_pairs.append((img_path, mask_path))

        if len(all_pairs) == 0:
            raise RuntimeError(f"No valid DRIVE image/vessel-mask pairs found in {train_dir}")

        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        n = len(all_pairs)
        n_train = int(round(n * split_ratios[0]))
        n_val = int(round(n * split_ratios[1]))

        if split == "train":
            self.samples = all_pairs[:n_train]
        elif split == "val":
            self.samples = all_pairs[n_train : n_train + n_val]
        else:
            self.samples = all_pairs[n_train + n_val :]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            image = TF.resize(
                image, self.image_size, interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask, self.image_size, interpolation=InterpolationMode.NEAREST
            )

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        X = TF.to_tensor(image)
        Y = (TF.to_tensor(mask) > 0.5).float()
        return X, Y
