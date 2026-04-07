"""
SE-DenseNet-3D for Schizophrenia Classification from Structural MRI
=====================================================================
2-channel input: [z-scored brain volume, GM probability map] @ MNI152 2mm (91x109x91)
Leave-one-site-out cross-validation across sites (DS 0,1,2,3,4,6,7)
Grad-CAM interpretability on final dense block at 5x7x5 resolution

Architecture: 3D DenseNet with SE + site-adversarial training
  - Initial conv: 2 -> 48 ch @ 23x28x23
  - Dense Block 1 (4 layers, k=16): 112 -> Transition -> 56 @ 11x14x11
  - Dense Block 2 (6 layers, k=16): 152 -> Transition -> 76 @ 5x7x5
  - Dense Block 3 (8 layers, k=16): 204 @ 5x7x5  <- Grad-CAM target
  - GAP -> 204
  - Classifier head: 204 -> 128 -> 1 (schiz/healthy)
  - Site adversary head: GRL -> 204 -> 64 -> num_sites (gradient reversal)

Improvements over base model:
  1. Site-adversarial training (gradient reversal layer)
     Forces features to be site-invariant by penalizing site-predictive
     representations. Alpha ramps 0->1 over training via schedule.
  2. 3D Cutout augmentation
     Zeros a random 10x10x10 cube from both channels. Forces model to
     distribute attention across multiple brain regions. Simulates
     realistic signal dropout near air-tissue boundaries.
  3. Learning rate warmup
     Linear ramp from lr/50 to lr over first 5 epochs. Stabilizes
     BN statistics and initial conv weights before full lr kicks in.

Total params: ~730K (main) + ~13K (site adversary) = ~743K
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, f1_score
)
import csv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("schiz-model")


# ===========================================================================
# MODEL COMPONENTS
# ===========================================================================

class SqueezeExcitation3D(nn.Module):
    """3D channel attention via squeeze-and-excitation."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        z = self.squeeze(x).view(b, c)
        s = self.excitation(z).view(b, c, 1, 1, 1)
        return x * s


class DenseLayer3D(nn.Module):
    """BN -> ReLU -> 1x1 Conv (bottleneck 4k) -> BN -> ReLU -> 3x3 Conv -> Dropout"""
    def __init__(self, in_channels: int, growth_rate: int, dropout: float = 0.2):
        super().__init__()
        bn_ch = 4 * growth_rate
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, bn_ch, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_ch)
        self.conv2 = nn.Conv3d(bn_ch, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock3D(nn.Module):
    """Stack of DenseLayer3D with dense connections + SE after."""
    def __init__(self, in_channels, num_layers, growth_rate, dropout=0.2, se_reduction=8):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer3D(ch, growth_rate, dropout))
            ch += growth_rate
        self.se = SqueezeExcitation3D(ch, reduction=se_reduction)
        self.out_channels = ch

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.se(x)


class Transition3D(nn.Module):
    """BN -> 1x1 Conv (compress) -> AvgPool 2x2x2."""
    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        out_ch = int(in_channels * compression)
        self.norm = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.out_channels = out_ch

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x), inplace=True)))


# ===========================================================================
# GRADIENT REVERSAL LAYER (for site-adversarial training)
# ===========================================================================

class GradientReversalFn(torch.autograd.Function):
    """
    Forward: identity.
    Backward: negate gradients and scale by alpha.

    This is the core trick: the site classifier's loss produces gradients
    that would normally make features MORE site-predictive. By flipping
    the sign, we make the feature extractor LESS site-predictive instead.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SiteAdversary(nn.Module):
    """
    Auxiliary classifier that predicts acquisition site from shared features.

    Gradient reversal ensures the shared feature extractor learns to
    produce site-INVARIANT representations. The adversary tries to
    predict site; the feature extractor tries to fool it.

    Architecture: GRL -> FC(204->64) -> ReLU -> Dropout -> FC(64->num_sites)
    ~13K params — trivial overhead.

    Alpha schedule: ramps from 0 to 1 during training.
      - Early epochs (alpha~0): adversary gradients are suppressed,
        model focuses on learning schiz classification features.
      - Later epochs (alpha~1): adversary gradients at full strength,
        model is forced to suppress site-specific features.
    """
    def __init__(self, in_features: int, num_sites: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_sites),
        )

    def forward(self, features, alpha: float = 1.0):
        reversed_features = GradientReversalFn.apply(features, alpha)
        return self.head(reversed_features)

    @staticmethod
    def get_alpha(epoch: int, max_epochs: int) -> float:
        """
        Sigmoid ramp schedule for alpha.
        Starts near 0, reaches ~1 by 2/3 through training.
        Smoother than linear ramp — avoids sudden gradient reversal shock.
        """
        p = epoch / max_epochs
        return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ===========================================================================
# MAIN MODEL
# ===========================================================================

class SEDenseNet3D(nn.Module):
    """
    3D DenseNet with SE blocks + optional site adversary.

    Input:  (B, 2, 91, 109, 91)
    Output: (B, 1) logit, and optionally (B, num_sites) site logits

    Grad-CAM hooks into Block 2 output at 5x7x5 = 175 spatial positions.
    """

    def __init__(
        self,
        in_channels: int = 2,
        init_features: int = 48,
        growth_rate: int = 16,
        block_config: Tuple[int, ...] = (4, 6, 8),
        compression: float = 0.5,
        dropout: float = 0.2,
        se_reduction: int = 8,
        classifier_dropout: float = 0.5,
        num_sites: int = 0,
    ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        ch = init_features

        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(ch, num_layers, growth_rate, dropout, se_reduction)
            self.blocks.append(block)
            ch = block.out_channels
            if i < len(block_config) - 1:
                trans = Transition3D(ch, compression)
                self.transitions.append(trans)
                ch = trans.out_channels

        self.final_norm = nn.BatchNorm3d(ch)
        self.feature_channels = ch  # 204

        # Disease classifier
        self.classifier = nn.Sequential(
            nn.Linear(ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1),
        )

        # Site adversary (optional — only created if num_sites > 0)
        self.site_adversary = SiteAdversary(ch, num_sites) if num_sites > 0 else None

        self._init_weights()

        # Grad-CAM state
        self._gc_feats = {}
        self._gc_grads = {}
        self._gc_hooks = []

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha: float = 0.0):
        """
        Args:
            x: (B, 2, 91, 109, 91)
            alpha: gradient reversal strength for site adversary.
                   0.0 = no reversal (early training), 1.0 = full reversal.

        Returns:
            disease_logits: (B, 1)
            site_logits: (B, num_sites) or None if no adversary
        """
        out = self.initial(x)

        t = 0
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.blocks) - 1:
                out = self.transitions[t](out)
                t += 1

        out = F.relu(self.final_norm(out), inplace=True)
        features = F.adaptive_avg_pool3d(out, 1).view(out.size(0), -1)  # (B, 204)

        disease_logits = self.classifier(features)

        site_logits = None
        if self.site_adversary is not None:
            site_logits = self.site_adversary(features, alpha)

        return disease_logits, site_logits

    # --- Grad-CAM -----------------------------------------------------------

    def enable_gradcam(self, target_blocks: Tuple[int, ...] = (2,)):
        """Hook dense blocks for Grad-CAM. Default: block 2 (5x7x5, 204ch)."""
        self.disable_gradcam()
        self._gc_targets = target_blocks

        for block_idx in target_blocks:
            key = f"block_{block_idx}"

            def make_hook(k):
                def hook_fn(module, inp, out):
                    self._gc_feats[k] = out
                    out.register_hook(lambda g, k=k: self._gc_grads.update({k: g.detach()}))
                return hook_fn

            hook = self.blocks[block_idx].register_forward_hook(make_hook(key))
            self._gc_hooks.append(hook)

    def disable_gradcam(self):
        for h in self._gc_hooks:
            h.remove()
        self._gc_feats = {}
        self._gc_grads = {}
        self._gc_hooks = []
        self._gc_targets = ()

    def compute_gradcam(self, output_size: Tuple[int, int, int] = (91, 109, 91)) -> np.ndarray:
        """Compute Grad-CAM. Multi-scale if multiple blocks hooked."""
        cams = []
        for block_idx in self._gc_targets:
            key = f"block_{block_idx}"
            grads = self._gc_grads[key]
            feats = self._gc_feats[key]
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=output_size, mode="trilinear", align_corners=False)
            cams.append(cam)

        cam = torch.stack(cams).mean(dim=0)
        cam = cam.detach().squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)
        return cam

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# AUGMENTATION
# ===========================================================================

class RandomFlip3D:
    """L-R flip along sagittal axis."""
    def __init__(self, p=0.5): self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:
            return np.flip(x, axis=-1).copy()
        return x

class RandomNoise3D:
    """Gaussian noise on brain channel only (ch 0)."""
    def __init__(self, std_range=(0.0, 0.03), p=0.4):
        self.std_range, self.p = std_range, p
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = x.copy()
            x[0] += np.random.randn(*x[0].shape).astype(np.float32) * np.random.uniform(*self.std_range)
        return x

class RandomIntensityScale:
    """Gain variation on brain channel."""
    def __init__(self, scale_range=(0.92, 1.08), p=0.3):
        self.scale_range, self.p = scale_range, p
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = x.copy()
            x[0] *= np.random.uniform(*self.scale_range)
        return x

class RandomBrightness3D:
    """Additive shift on brain channel."""
    def __init__(self, max_shift=0.08, p=0.3):
        self.max_shift, self.p = max_shift, p
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = x.copy()
            x[0] += np.random.uniform(-self.max_shift, self.max_shift)
        return x

class RandomCutout3D:
    """
    Zero out a random contiguous cube from BOTH channels.

    Size 10x10x10 at 2mm resolution = 20x20x20mm cube.
    This is large enough to occlude a subcortical nucleus but small
    enough to leave most of the brain visible.

    Effect: forces the model to distribute its decision-making across
    multiple brain regions. Without cutout, the model can concentrate
    100% of its attention on one spot (e.g. left hippocampus). With
    cutout, that spot is sometimes blacked out during training, so the
    model must also learn from thalamus, ventricles, caudate, etc.

    Also simulates real signal dropout near air-tissue boundaries
    (sinuses, temporal poles, ear canals) — a realistic artifact.
    """
    def __init__(self, cube_size: int = 10, p: float = 0.5):
        self.cube_size = cube_size
        self.p = p

    def __call__(self, x):
        # x shape: (2, D, H, W) = (2, 91, 109, 91)
        if np.random.rand() < self.p:
            x = x.copy()
            _, d, h, w = x.shape
            s = self.cube_size
            # Random position (clamped so cube fits within volume)
            cd = np.random.randint(0, max(d - s, 1))
            ch = np.random.randint(0, max(h - s, 1))
            cw = np.random.randint(0, max(w - s, 1))
            # Zero both channels — brain volume AND GM map
            x[:, cd:cd+s, ch:ch+s, cw:cw+s] = 0.0
        return x

class Compose3D:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms: x = t(x)
        return x

def get_train_augmentation():
    return Compose3D([
        RandomFlip3D(p=0.5),
        RandomNoise3D(std_range=(0.0, 0.03), p=0.4),
        RandomIntensityScale(scale_range=(0.92, 1.08), p=0.3),
        RandomBrightness3D(max_shift=0.08, p=0.3),
        RandomCutout3D(cube_size=10, p=0.5),  # NEW: 3D cutout
    ])


# ===========================================================================
# DATASET
# ===========================================================================

class SchizMRIDataset(Dataset):
    """Loads 2-channel 3D: preprocessed brain + GM map."""
    def __init__(self, entries: List[Dict], transform=None):
        self.entries = entries
        self.transform = transform
        import nibabel as nib
        self.nib = nib

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        bn, pp = e["basename"], e["pp_dir"]

        brain = self.nib.load(os.path.join(pp, f"{bn}_preprocessed.nii.gz")).get_fdata(dtype=np.float32)
        gm = self.nib.load(os.path.join(pp, f"{bn}_gm.nii.gz")).get_fdata(dtype=np.float32)

        vol = np.stack([brain, gm], axis=0)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform:
            vol = self.transform(vol)

        return (
            torch.from_numpy(vol) if isinstance(vol, np.ndarray) else vol,
            torch.tensor(e["label"], dtype=torch.float32),
            torch.tensor(e["dscode"], dtype=torch.long),  # now tensor for site loss
            bn,
        )


# ===========================================================================
# CSV PARSING
# ===========================================================================

def win_to_wsl(path: str) -> str:
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        return f"/mnt/{path[0].lower()}{path[2:]}"
    return path

def build_entries_from_csv(csv_path: str) -> List[Dict]:
    entries, skipped = [], 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            grp = row["grp"].strip()
            if grp not in ("1", "3"):
                continue
            filename = row["filename"].strip().strip("\r")
            filepath = row["filepath"].strip().strip("\r")
            label = int(row["label"].strip())
            dscode = int(row["dscode"].strip())
            basename = filename.replace(".nii.gz", "")

            wsl_path = win_to_wsl(filepath)
            nifti_dir = os.path.dirname(wsl_path)
            scans_dir = os.path.dirname(nifti_dir)
            pp_dir = os.path.join(scans_dir, "preprocessed", f"PP_{basename}")

            if not os.path.exists(os.path.join(pp_dir, f"{basename}_preprocessed.nii.gz")):
                skipped += 1
                continue

            entries.append({"basename": basename, "pp_dir": pp_dir,
                            "label": label, "dscode": dscode})

    log.info(f"Found {len(entries)} preprocessed scans, skipped {skipped}")
    log.info(f"Labels: {dict(Counter(e['label'] for e in entries))}")
    log.info(f"Sites:  {dict(sorted(Counter(e['dscode'] for e in entries).items()))}")
    return entries


# ===========================================================================
# TRAINING UTILITIES
# ===========================================================================

def get_class_weights(entries):
    n_pos = sum(e["label"] for e in entries)
    n_neg = len(entries) - n_pos
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

def get_weighted_sampler(entries):
    labels = [e["label"] for e in entries]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    w = {1: len(labels) / (2 * max(n_pos, 1)), 0: len(labels) / (2 * max(n_neg, 1))}
    return WeightedRandomSampler([w[l] for l in labels], len(entries), replacement=True)

def build_site_mapping(entries: List[Dict]) -> Dict[int, int]:
    """
    Map original dscode values to contiguous indices for CrossEntropyLoss.
    DS codes might be [0,1,2,3,4,6,7] — need mapping to [0,1,2,3,4,5,6].
    """
    unique_sites = sorted(set(e["dscode"] for e in entries))
    return {ds: idx for idx, ds in enumerate(unique_sites)}

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, mode="max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter, self.best_state = 0, None

    def step(self, metric, model):
        improved = (metric > self.best + self.min_delta if self.mode == "max"
                    else metric < self.best - self.min_delta)
        if improved:
            self.best, self.counter = metric, 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.02):
    """
    Linear warmup for warmup_epochs, then cosine annealing to min_lr.

    Warmup: lr ramps from lr*min_lr_ratio to lr linearly.
    This stabilizes BatchNorm statistics and initial conv weights
    before full learning rate kicks in. Critical with batch_size=4
    where BN mean/var estimates are noisy.

    After warmup: cosine decay with warm restarts (T_0=15, T_mult=2).
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear ramp from min_lr_ratio to 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * (epoch / warmup_epochs)
        else:
            # Cosine annealing with warm restarts
            adjusted = epoch - warmup_epochs
            t_0 = 15
            t_mult = 2
            # Find which cycle we're in
            cycle_len = t_0
            cycle_start = 0
            while adjusted >= cycle_start + cycle_len:
                cycle_start += cycle_len
                cycle_len *= t_mult
            progress = (adjusted - cycle_start) / cycle_len
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# TRAIN / EVALUATE
# ===========================================================================

def train_one_epoch(model, loader, disease_criterion, site_criterion,
                    optimizer, device, epoch, max_epochs, site_map,
                    adversarial_lambda=0.1, scaler=None):
    """
    Train one epoch with disease classification + site adversarial loss.
    Supports mixed precision (FP16) via torch.amp when scaler is provided.

    Total loss = disease_loss + adversarial_lambda * site_loss

    The site_loss gradients get REVERSED before reaching the feature
    extractor (via GradientReversalFn), so minimizing site_loss in the
    adversary head simultaneously makes the feature extractor WORSE
    at predicting site — i.e. more site-invariant.
    """
    model.train()
    total_loss, total_disease_loss, total_site_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0
    use_amp = scaler is not None

    alpha = SiteAdversary.get_alpha(epoch, max_epochs)

    for volumes, labels, dscodes, _ in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Map dscodes to contiguous site indices
        site_targets = torch.tensor(
            [site_map[ds.item()] for ds in dscodes],
            dtype=torch.long, device=device
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision: forward pass in FP16 on Tensor Cores
        # Convolutions and matmuls run in FP16 (~2x faster on RTX 3060)
        # Loss computation stays FP32 for numerical stability
        with torch.amp.autocast("cuda", enabled=use_amp):
            disease_logits, site_logits = model(volumes, alpha=alpha)
            disease_logits = disease_logits.squeeze(1)

            d_loss = disease_criterion(disease_logits, labels)

            if site_logits is not None:
                s_loss = site_criterion(site_logits, site_targets)
                loss = d_loss + adversarial_lambda * s_loss
                total_site_loss += s_loss.item() * volumes.size(0)
            else:
                loss = d_loss

        # Mixed precision: GradScaler prevents FP16 gradient underflow
        # Scales loss up before backward, scales gradients down before step
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # unscale before clip_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * volumes.size(0)
        total_disease_loss += d_loss.item() * volumes.size(0)
        correct += ((torch.sigmoid(disease_logits) > 0.5).float() == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "disease_loss": total_disease_loss / total,
        "site_loss": total_site_loss / total,
        "accuracy": correct / total,
        "alpha": alpha,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, all_probs, all_labels, all_names = 0.0, [], [], []
    for volumes, labels, _, names in loader:
        volumes, labels = volumes.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            disease_logits, _ = model(volumes, alpha=0.0)
            disease_logits = disease_logits.squeeze(1)
            total_loss += criterion(disease_logits, labels).item() * volumes.size(0)
        all_probs.extend(torch.sigmoid(disease_logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_names.extend(names)

    probs, labels = np.array(all_probs), np.array(all_labels)
    preds = (probs > 0.5).astype(float)
    try: auc = roc_auc_score(labels, probs)
    except ValueError: auc = 0.0

    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, preds),
        "auc": auc,
        "sensitivity": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "probs": probs, "labels": labels, "basenames": all_names,
    }


def run_fold(fold_idx, test_site, train_entries, test_entries, args, device):
    log.info(f"\n{'='*60}")
    log.info(f"FOLD {fold_idx}: Test site = DS{test_site}")
    log.info(f"  Train: {len(train_entries)} | Test: {len(test_entries)}")
    log.info(f"  Train labels: {dict(Counter(e['label'] for e in train_entries))}")
    log.info(f"  Test  labels: {dict(Counter(e['label'] for e in test_entries))}")

    # Build site mapping from TRAINING sites only (test site excluded)
    site_map = build_site_mapping(train_entries)
    num_train_sites = len(site_map)
    log.info(f"  Training sites: {site_map} ({num_train_sites} sites)")

    # Validation split: 15% of train, stratified
    rng = np.random.RandomState(42 + fold_idx)
    pos_idx = [i for i, e in enumerate(train_entries) if e["label"] == 1]
    neg_idx = [i for i, e in enumerate(train_entries) if e["label"] == 0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)

    val_n_pos = max(1, int(len(pos_idx) * 0.15))
    val_n_neg = max(1, int(len(neg_idx) * 0.15))
    val_set = set(pos_idx[:val_n_pos] + neg_idx[:val_n_neg])

    train_sub = [train_entries[i] for i in range(len(train_entries)) if i not in val_set]
    val_sub = [train_entries[i] for i in val_set]
    log.info(f"  Train sub: {len(train_sub)} | Val sub: {len(val_sub)}")

    train_ds = SchizMRIDataset(train_sub, transform=get_train_augmentation())
    val_ds = SchizMRIDataset(val_sub)
    test_ds = SchizMRIDataset(test_entries)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=get_weighted_sampler(train_sub),
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0,
                              prefetch_factor=2 if args.workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0,
                            prefetch_factor=2 if args.workers > 0 else None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0,
                             prefetch_factor=2 if args.workers > 0 else None)

    # Model with site adversary
    model = SEDenseNet3D(
        in_channels=2, init_features=48, growth_rate=16,
        block_config=(4, 6, 8), compression=0.5,
        dropout=0.2, se_reduction=8, classifier_dropout=0.5,
        num_sites=num_train_sites,
    ).to(device)

    if fold_idx == 0:
        log.info(f"  Parameters: {model.count_parameters():,}")

    # --- GPU optimizations ---

    # cuDNN benchmark: since input size is FIXED (91x109x91 every batch),
    # cuDNN benchmarks conv algorithms on first batch and caches the fastest.
    # ~10-15% speedup on 3D convolutions. Only set once globally.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # torch.compile: JIT-compiles the model, fuses ops, reduces kernel
    # launch overhead. First epoch is slower (compilation), subsequent
    # epochs 10-30% faster. Requires PyTorch 2.0+.
    if args.compile and hasattr(torch, "compile"):
        log.info("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Mixed precision (AMP): forward pass runs in FP16 on Tensor Cores
    # (~2x faster convolutions on RTX 3060 Ampere), backward pass uses
    # GradScaler to prevent FP16 gradient underflow. ~40% VRAM savings.
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        log.info("  Mixed precision (FP16) enabled")

    # Loss functions
    pos_weight = get_class_weights(train_sub).to(device)
    disease_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    site_criterion = nn.CrossEntropyLoss()

    # Optimizer (includes adversary params — they need to get better at
    # site prediction so the gradient reversal has something to fight against)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # LR schedule: warmup 5 epochs then cosine annealing with restarts
    scheduler = get_warmup_cosine_scheduler(
        optimizer, warmup_epochs=5, total_epochs=args.epochs, min_lr_ratio=0.02
    )

    early_stop = EarlyStopping(patience=args.patience, min_delta=0.002, mode="max")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, disease_criterion, site_criterion,
            optimizer, device, epoch, args.epochs, site_map,
            adversarial_lambda=args.adv_lambda, scaler=scaler,
        )
        scheduler.step()
        val_m = evaluate(model, val_loader, disease_criterion, device, use_amp=use_amp)
        dt = time.time() - t0

        if epoch % 5 == 0 or epoch <= 3:
            log.info(
                f"  E{epoch:3d} | d_loss={tr['disease_loss']:.4f} "
                f"s_loss={tr['site_loss']:.4f} alpha={tr['alpha']:.2f} "
                f"tr_acc={tr['accuracy']:.3f} | "
                f"val_auc={val_m['auc']:.3f} val_acc={val_m['accuracy']:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e} {dt:.1f}s"
            )

        if early_stop.step(val_m["auc"], model):
            log.info(f"  Early stop at epoch {epoch}, best val AUC={early_stop.best:.4f}")
            break

    early_stop.restore_best(model)
    model.to(device)
    test_m = evaluate(model, test_loader, disease_criterion, device, use_amp=use_amp)
    log.info(f"  TEST DS{test_site}: AUC={test_m['auc']:.3f} Acc={test_m['accuracy']:.3f} "
             f"Sens={test_m['sensitivity']:.3f} Spec={test_m['specificity']:.3f} "
             f"F1={test_m['f1']:.3f}")

    save_dir = Path(args.output_dir) / f"fold_{fold_idx}_DS{test_site}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")

    pred_data = {
        "test_site": test_site, "basenames": test_m["basenames"],
        "labels": test_m["labels"].tolist(), "probs": test_m["probs"].tolist(),
        "metrics": {k: v for k, v in test_m.items()
                    if k not in ("probs", "labels", "basenames")},
    }
    with open(save_dir / "test_predictions.json", "w") as f:
        json.dump(pred_data, f, indent=2)

    return test_m


# ===========================================================================
# GRAD-CAM GENERATION
# ===========================================================================

def generate_gradcam(model_path, entries, device, output_dir, multi_scale=False,
                     num_sites=0):
    import nibabel as nib

    model = SEDenseNet3D(num_sites=num_sites)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    target = (1, 2) if multi_scale else (2,)
    model.enable_gradcam(target_blocks=target)
    log.info(f"Grad-CAM target blocks: {target}")

    schiz = [e for e in entries if e["label"] == 1]
    loader = DataLoader(SchizMRIDataset(schiz), batch_size=1, shuffle=False)

    cam_sum = np.zeros((91, 109, 91), dtype=np.float64)
    for i, (vol, _, _, bn) in enumerate(loader):
        vol = vol.to(device).requires_grad_(True)
        disease_logits, _ = model(vol, alpha=0.0)
        disease_logits.squeeze().backward()
        cam_sum += model.compute_gradcam()
        if (i + 1) % 50 == 0:
            log.info(f"  Grad-CAM: {i+1}/{len(schiz)}")

    avg = (cam_sum / len(schiz)).astype(np.float32)

    affine = np.array([[-2,0,0,90],[0,2,0,-126],[0,0,2,-72],[0,0,0,1]], dtype=np.float64)
    os.makedirs(output_dir, exist_ok=True)
    suffix = "multiscale" if multi_scale else "block2"
    out_path = os.path.join(output_dir, f"gradcam_schiz_avg_{suffix}.nii.gz")
    nib.save(nib.Nifti1Image(avg, affine), out_path)
    log.info(f"Saved Grad-CAM to {out_path}")
    model.disable_gradcam()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="SE-DenseNet-3D Schizophrenia Classification")
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", default="./runs")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--adv-lambda", type=float, default=0.1,
                   help="Weight for site adversarial loss. 0.1 is conservative. "
                        "Higher = stronger site suppression but may hurt disease accuracy.")
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--gradcam-model", type=str, default=None)
    p.add_argument("--gradcam-multiscale", action="store_true")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Mixed precision FP16 training (default: on). "
                        "~2x faster on RTX 3060 Tensor Cores, ~40%% less VRAM.")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Disable mixed precision (use FP32)")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for 10-30%% speedup (PyTorch 2.0+). "
                        "First epoch is slow due to compilation.")
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    log.info(f"Device: {device}")
    if device.type == "cpu":
        log.warning("CPU mode — expect ~5-10 min/epoch. Use CUDA if available.")

    entries = build_entries_from_csv(args.csv)
    if not entries:
        log.error("No preprocessed scans found. Run preprocess_from_csv.sh first.")
        sys.exit(1)

    if args.gradcam_model:
        # Need num_sites to reconstruct model architecture for loading weights
        all_sites = sorted(set(e["dscode"] for e in entries))
        generate_gradcam(args.gradcam_model, entries, device, args.output_dir,
                         multi_scale=args.gradcam_multiscale,
                         num_sites=len(all_sites) - 1)  # LOSO: one site excluded
        return

    sites = sorted(set(e["dscode"] for e in entries))
    log.info(f"LOSO sites: {sites}")
    all_metrics = []

    for fi, ts in enumerate(sites):
        if args.fold is not None and fi != args.fold:
            continue

        test_e = [e for e in entries if e["dscode"] == ts]
        train_e = [e for e in entries if e["dscode"] != ts]

        test_labels = set(e["label"] for e in test_e)
        if len(test_labels) < 2:
            log.warning(f"FOLD {fi}: DS{ts} single-class ({test_labels}). AUC undefined.")

        m = run_fold(fi, ts, train_e, test_e, args, device)
        all_metrics.append({"fold": fi, "test_site": ts, "n_test": len(test_e),
                            **{k: v for k, v in m.items()
                               if k not in ("probs", "labels", "basenames")}})

    if len(all_metrics) > 1:
        log.info(f"\n{'='*60}")
        log.info("LOSO CROSS-VALIDATION SUMMARY")
        log.info(f"{'='*60}")
        total_n = sum(m["n_test"] for m in all_metrics)
        for name in ["accuracy", "auc", "sensitivity", "specificity", "f1"]:
            vals = [m[name] for m in all_metrics]
            w = [m["n_test"] for m in all_metrics]
            wavg = sum(v * wi for v, wi in zip(vals, w)) / total_n
            log.info(f"  {name:12s}: weighted={wavg:.3f}  "
                     f"mean={np.mean(vals):.3f}+/-{np.std(vals):.3f}  "
                     f"per-fold={[f'{v:.3f}' for v in vals]}")

        with open(Path(args.output_dir) / "loso_summary.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

    if args.gradcam and args.fold is not None:
        mp = Path(args.output_dir) / f"fold_{args.fold}_DS{sites[args.fold]}" / "model.pt"
        if mp.exists():
            generate_gradcam(str(mp), entries, device,
                             str(Path(args.output_dir) / f"gradcam_fold_{args.fold}"),
                             multi_scale=args.gradcam_multiscale,
                             num_sites=len(sites) - 1)


if __name__ == "__main__":
    main()
