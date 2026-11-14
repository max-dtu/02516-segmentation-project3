import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 conv-BN-ReLU blocks preserving spatial dimensions."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4-level U-Net for binary image segmentation (works for H, W divisible by 16)."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c5)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x, trace_shapes=False):
        shapes = []

        def record(name, t):
            if trace_shapes:
                shapes.append((name, tuple(t.shape)))

        record("input", x)
        e1 = self.enc1(x)
        record("enc1", e1)
        p1 = self.pool1(e1)
        record("pool1", p1)

        e2 = self.enc2(p1)
        record("enc2", e2)
        p2 = self.pool2(e2)
        record("pool2", p2)

        e3 = self.enc3(p2)
        record("enc3", e3)
        p3 = self.pool3(e3)
        record("pool3", p3)

        e4 = self.enc4(p3)
        record("enc4", e4)
        p4 = self.pool4(e4)
        record("pool4", p4)

        b = self.bottleneck(p4)
        record("bottleneck", b)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        record("dec4", d4)

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        record("dec3", d3)

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        record("dec2", d2)

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        record("dec1", d1)

        out = self.head(d1)
        record("output", out)

        if trace_shapes:
            return out, shapes
        return out
