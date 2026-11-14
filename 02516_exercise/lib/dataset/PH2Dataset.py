import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

DEFAULT_PH2_PATH = "/dtu/datasets1/02516/PH2_Dataset_images"


class PH2Dataset(Dataset):
    """PyTorch Dataset for the PH2 skin lesion segmentation dataset.

    Expects directory layout:
        <data_path>/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp
        <data_path>/IMDxxx/IMDxxx_lesion/IMDxxx_lesion.bmp
    """

    def __init__(
        self,
        split="train",
        data_path=DEFAULT_PH2_PATH,
        image_size=(256, 256),
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

        # Collect all valid IMDxxx sample pairs sorted deterministically
        all_pairs = []
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"PH2 dataset directory not found: {data_path}")

        for case_id in sorted(os.listdir(data_path)):
            case_dir = os.path.join(data_path, case_id)
            if not os.path.isdir(case_dir):
                continue
            img_path = os.path.join(
                case_dir, f"{case_id}_Dermoscopic_Image", f"{case_id}.bmp"
            )
            mask_path = os.path.join(
                case_dir, f"{case_id}_lesion", f"{case_id}_lesion.bmp"
            )
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                all_pairs.append((img_path, mask_path))

        if len(all_pairs) == 0:
            raise RuntimeError(f"No valid PH2 image/mask pairs found in {data_path}")

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
