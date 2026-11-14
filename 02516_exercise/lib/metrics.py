import torch


def compute_confusion_counts(logits, targets, threshold=0.5):
    """Compute TP, FP, TN, FN counts from raw model logits and binary targets."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).bool()
    targs = (targets >= 0.5).bool()

    tp = (preds & targs).sum().item()
    fp = (preds & ~targs).sum().item()
    tn = (~preds & ~targs).sum().item()
    fn = (~preds & targs).sum().item()
    return tp, fp, tn, fn


def metrics_from_counts(tp, fp, tn, fn, eps=1e-7):
    """Compute Dice, IoU, Accuracy, Sensitivity, and Specificity from confusion counts."""
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


@torch.no_grad()
def evaluate_model(model, loader, device, loss_fn=None, threshold=0.5):
    """Evaluate a segmentation model on a DataLoader across all 5 metrics."""
    model.eval()
    total_tp = total_fp = total_tn = total_fn = 0
    total_loss = 0.0

    for x_batch, y_true in loader:
        x_batch = x_batch.to(device)
        y_true = y_true.to(device)
        logits = model(x_batch)

        if loss_fn is not None:
            total_loss += loss_fn(logits, y_true).item()

        tp, fp, tn, fn = compute_confusion_counts(logits, y_true, threshold=threshold)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    metrics = metrics_from_counts(total_tp, total_fp, total_tn, total_fn)
    if loss_fn is not None and len(loader) > 0:
        metrics["loss"] = total_loss / len(loader)
    return metrics
