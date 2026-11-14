import argparse
import json
import os
from time import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.dataset.PH2Dataset import PH2Dataset
from lib.dataset.PH2WeakDataset import PH2WeakDataset
from lib.losses import PointBCELoss
from lib.metrics import evaluate_model
from lib.model.UNetModel import UNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_metrics(m):
    return (
        f"Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | "
        f"Acc={m['accuracy']:.4f} | Sens={m['sensitivity']:.4f} | Spec={m['specificity']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Weakly-supervised point-click segmentation on PH2")
    parser.add_argument("--n-pos", type=int, default=10)
    parser.add_argument("--n-neg", type=int, default=10)
    parser.add_argument("--strategy", choices=["uniform", "interior"], default="uniform")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()

    weak_kwargs = {
        "image_size": (args.image_size, args.image_size),
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "strategy": args.strategy,
    }
    full_kwargs = {"image_size": (args.image_size, args.image_size)}
    if args.data_path:
        weak_kwargs["data_path"] = args.data_path
        full_kwargs["data_path"] = args.data_path

    trainset = PH2WeakDataset(**weak_kwargs)
    valset = PH2Dataset(split="val", **full_kwargs)
    testset = PH2Dataset(split="test", **full_kwargs)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet().to(device)
    loss_fn = PointBCELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"ph2_weak_{args.strategy}_{args.n_pos}p_{args.n_neg}n"
    ckpt_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        tic = time()
        model.train()
        train_loss = 0.0
        for x_batch, click_target, click_mask in train_loader:
            x_batch = x_batch.to(device)
            click_target = click_target.to(device)
            click_mask = click_mask.to(device)

            opt.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, click_target, click_mask)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_metrics = evaluate_model(model, val_loader, device)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} ({time() - tic:.1f}s) | "
                f"train_loss={train_loss:.4f} | {format_metrics(val_metrics)}"
            )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    results = {
        "run_name": run_name,
        "args": vars(args),
        "val": evaluate_model(model, val_loader, device),
        "test": evaluate_model(model, test_loader, device),
    }

    print(f"\n=== Best Checkpoint Results ({run_name}) ===")
    for split in ("val", "test"):
        print(f"  {split.upper():5s}: {format_metrics(results[split])}")

    json_path = os.path.join(args.save_dir, f"{run_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
