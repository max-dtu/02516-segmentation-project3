import os
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

from lib.dataset.DRIVEDataset import DRIVEDataset
from lib.dataset.PH2Dataset import PH2Dataset
from lib.model.EncDecModel import EncDec
from lib.model.UNetModel import UNet


def load_model(model_cls, ckpt_path, device):
    model = model_cls().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, x, device, threshold=0.5):
    logits = model(x.unsqueeze(0).to(device))
    prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu()
    mask = (prob >= threshold).float()
    return TF.to_pil_image(mask).convert("RGB")


def make_grid(dataset_name, dataset, models, out_path, device, sample_indices=(0, 1)):
    cols = ["Input", "Ground Truth"] + [name for name, _ in models]
    cell_w, cell_h = 256, 256
    header_h = 28
    pad = 4

    n_rows = len(sample_indices)
    n_cols = len(cols)
    canvas_w = n_cols * cell_w + (n_cols + 1) * pad
    canvas_h = header_h + n_rows * cell_h + (n_rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for c_idx, title in enumerate(cols):
        x0 = pad + c_idx * (cell_w + pad) + 8
        draw.text((x0, 6), title, fill=(0, 0, 0))

    for r_idx, sample_idx in enumerate(sample_indices):
        x, y = dataset[sample_idx]
        inp_img = TF.to_pil_image(x).resize((cell_w, cell_h), Image.BILINEAR)
        gt_img = TF.to_pil_image(y).convert("RGB").resize((cell_w, cell_h), Image.NEAREST)

        row_imgs = [inp_img, gt_img]
        for _, model in models:
            pred_img = predict_mask(model, x, device).resize((cell_w, cell_h), Image.NEAREST)
            row_imgs.append(pred_img)

        y0 = header_h + pad + r_idx * (cell_h + pad)
        for c_idx, img in enumerate(row_imgs):
            x0 = pad + c_idx * (cell_w + pad)
            canvas.paste(img, (x0, y0))

    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.size})")


def main():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PH2
    ph2_test = PH2Dataset(split="test", image_size=(256, 256))
    ph2_models = [
        ("EncDec (BCE)", load_model(EncDec, "outputs/ph2_encdec_bce_best.pt", device)),
        ("U-Net (BCE)", load_model(UNet, "outputs/ph2_unet_bce_best.pt", device)),
        ("U-Net (Focal)", load_model(UNet, "outputs/ph2_unet_focal_best.pt", device)),
        ("U-Net (WBCE)", load_model(UNet, "outputs/ph2_unet_wbce_best.pt", device)),
    ]
    make_grid("ph2", ph2_test, ph2_models, "outputs/qualitative_ph2.png", device, sample_indices=(0, 1))

    # DRIVE
    drive_test = DRIVEDataset(split="test", image_size=(512, 512))
    drive_models = [
        ("EncDec (BCE)", load_model(EncDec, "outputs/drive_encdec_bce_best.pt", device)),
        ("U-Net (BCE)", load_model(UNet, "outputs/drive_unet_bce_best.pt", device)),
        ("U-Net (Focal)", load_model(UNet, "outputs/drive_unet_focal_best.pt", device)),
        ("U-Net (WBCE)", load_model(UNet, "outputs/drive_unet_wbce_best.pt", device)),
    ]
    make_grid("drive", drive_test, drive_models, "outputs/qualitative_drive.png", device, sample_indices=(0, 1))


if __name__ == "__main__":
    main()
