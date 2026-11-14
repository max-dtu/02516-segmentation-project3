import argparse
import json
import os
from time import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.dataset.DRIVEDataset import DRIVEDataset
from lib.dataset.PH2Dataset import PH2Dataset
from lib.losses import BCELoss, FocalLoss, WeightedBCELoss
from lib.metrics import evaluate_model
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_datasets(dataset_name, image_size, data_path=None):
    if dataset_name == "ph2":
        size = image_size or 256
        kwargs = {"image_size": (size, size)}
        if data_path:
            kwargs["data_path"] = data_path
        return (
            PH2Dataset(split="train", **kwargs),
            PH2Dataset(split="val", **kwargs),
            PH2Dataset(split="test", **kwargs),
        )
    if dataset_name == "drive":
        size = image_size or 512
        kwargs = {"image_size": (size, size)}
        if data_path:
            kwargs["data_path"] = data_path
        return (
            DRIVEDataset(split="train", **kwargs),
            DRIVEDataset(split="val", **kwargs),
            DRIVEDataset(split="test", **kwargs),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_model(model_name):
    if model_name == "encdec":
        return EncDec()
    if model_name == "unet":
        return UNet()
    raise ValueError(f"Unknown model: {model_name}")


def build_loss(loss_name, dataset_name, pos_weight=None):
    if loss_name == "bce":
        return BCELoss()
    if loss_name == "focal":
        return FocalLoss(gamma=2.0)
    if loss_name == "wbce":
        if pos_weight is None:
            pos_weight = 2.0 if dataset_name == "ph2" else 8.0
        return WeightedBCELoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {loss_name}")


def format_metrics(m):
    return (
        f"Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | "
        f"Acc={m['accuracy']:.4f} | Sens={m['sensitivity']:.4f} | Spec={m['specificity']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate segmentation models")
    parser.add_argument("--dataset", choices=["ph2", "drive"], required=True)
    parser.add_argument("--model", choices=["encdec", "unet"], default="encdec")
    parser.add_argument("--loss", choices=["bce", "focal", "wbce"], default="bce")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    trainset, valset, testset = build_datasets(args.dataset, args.image_size, args.data_path)
    print(f"Dataset={args.dataset} | train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model).to(device)
    loss_fn = build_loss(args.loss, args.dataset, args.pos_weight).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{args.dataset}_{args.model}_{args.loss}"
    ckpt_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        tic = time()
        model.train()
        train_loss = 0.0
        for x_batch, y_true in train_loader:
            x_batch = x_batch.to(device)
            y_true = y_true.to(device)

            opt.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_true)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_metrics = evaluate_model(model, val_loader, device, loss_fn=loss_fn)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} ({time() - tic:.1f}s) | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"{format_metrics(val_metrics)}"
        )

    # Final evaluation on best checkpoint across all splits
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    results = {
        "run_name": run_name,
        "args": vars(args),
        "train": evaluate_model(model, train_loader, device, loss_fn=loss_fn),
        "val": evaluate_model(model, val_loader, device, loss_fn=loss_fn),
        "test": evaluate_model(model, test_loader, device, loss_fn=loss_fn),
    }

    print(f"\n=== Best Checkpoint Results ({run_name}) ===")
    for split in ("train", "val", "test"):
        print(f"  {split.upper():5s}: {format_metrics(results[split])}")

    json_path = os.path.join(args.save_dir, f"{run_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path} and metrics to {json_path}")


if __name__ == "__main__":
    main()
