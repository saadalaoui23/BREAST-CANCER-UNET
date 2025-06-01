#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRAIN_CMMD_256_UNET_SUBSET.PY — usage restreint d’images pour aller plus vite

• On ne prend qu’un sous‐ensemble des patients au lieu de tout le dataset
  (MAX_TRAIN_PATIENTS, MAX_VAL_PATIENTS)
• Résolution 256×256 (préprocessing doit avoir généré ces PNG)
• UNet léger + AMP + accumulations
"""

# ─── CONFIG (à ajuster pour accélérer davantage) ───────────────────
DATA_ROOT           = r"C:\Users\PC\Desktop\dataset_final_pp"
VAL_SPLIT           = 0.20
SEED                = 42

# on ne conserve qu’un sous‐ensemble de patients
MAX_TRAIN_PATIENTS  = 200   # au lieu de ~1 420
MAX_VAL_PATIENTS    = 50    # au lieu de ~355

BATCH_SIZE          = 4     # batch de 4 en VRAM
ACCUM_STEPS         = 2     # batch effectif = 8
NUM_WORKERS         = max(1, __import__("os").cpu_count() - 1)
MAX_EPOCHS          = 60

LR                  = 1e-4
WEIGHT_DECAY        = 1e-5
OUTPUT_DIR          = "./checkpoints_256_unet_subset"
# ───────────────────────────────────────────────────────────────────

import os, cv2, random, time, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

# on supprime les warnings d’AMP et d’Albumentations
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─── AMP compat (torch ≥2.1 ou torch.cuda.amp) ──────────────────────
try:
    from torch import amp as _amp
    scaler = _amp.GradScaler()
    def autocast(): return _amp.autocast(device_type="cuda")
except (ImportError, AttributeError):
    from torch.cuda import amp as _amp
    scaler = _amp.GradScaler()
    def autocast(): return _amp.autocast()

# ─── DATASET ────────────────────────────────────────────────────────
class CMMDataset(Dataset):
    """
    Parcourt les PNG 256×256 d’un sous‐ensemble de patients.
    """
    def __init__(self, root_dir, patient_ids, transforms=None):
        self.samples, self.tf = [], transforms
        root = Path(root_dir)
        for pid in patient_ids:
            p_dir = root / pid
            if not p_dir.is_dir():
                continue
            for img_p in p_dir.glob("image_*.png"):
                view = img_p.stem.split("image_")[1]
                mask_p = p_dir / f"mask_{view}.png"
                if mask_p.exists():
                    self.samples.append((str(img_p), str(mask_p)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]
        # chargement en 256×256 (comme préprocessed)
        img  = cv2.imread(img_p,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        img  = cv2.merge([img, img, img])
        mask = (mask > 127).astype(np.uint8)

        if self.tf:
            aug = self.tf(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        img  = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ─── MÉTRIQUES ──────────────────────────────────────────────────────
def dice(pred, tgt, eps=1e-7):
    pred = (pred > .5).float()
    tgt  = tgt.float()
    inter = (pred * tgt).sum((2,3))
    union = pred.sum((2,3)) + tgt.sum((2,3))
    return ((2*inter + eps) / (union + eps)).mean()

def iou(pred, tgt, eps=1e-7):
    pred = (pred > .5).float()
    tgt  = tgt.float()
    inter = (pred * tgt).sum((2,3))
    union = pred.sum((2,3)) + tgt.sum((2,3)) - inter
    return ((inter + eps) / (union + eps)).mean()

# ─── FONCTION PRINCIPALE ────────────────────────────────────────────
def main():
    # 1) Récupère tous les patients
    all_patients = [d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir()]
    random.shuffle(all_patients)

    # 2) On instancie le split classique patient-level
    n_val = int(len(all_patients) * VAL_SPLIT)
    val_list_total   = all_patients[:n_val]
    train_list_total = all_patients[n_val:]

    # 3) On réduit à un sous-ensemble limité
    train_pat = train_list_total[:MAX_TRAIN_PATIENTS]
    val_pat   = val_list_total[:MAX_VAL_PATIENTS]
    print(f">> Utilisation restreinte : train {len(train_pat)} patients / val {len(val_pat)} patients")

    # 4) Transformations légères (flip uniquement)
    train_tf = A.Compose([A.HorizontalFlip(p=0.5)])
    ds_train = CMMDataset(DATA_ROOT, train_pat, train_tf)
    ds_val   = CMMDataset(DATA_ROOT, val_pat, transforms=None)

    # 5) DataLoader avec prefetch_factor si possible
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4

    dl_train = DataLoader(ds_train, shuffle=True,  **loader_kwargs)
    val_loader_kwargs = dict(loader_kwargs)
    val_loader_kwargs["shuffle"] = False
    dl_val = DataLoader(ds_val, **val_loader_kwargs)

    print(f">> Échantillons : train {len(ds_train)} images / val {len(ds_val)} images")

    # 6) Modèle UNet léger
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss  = nn.BCEWithLogitsLoss()
    loss_fn   = lambda p, t: dice_loss(p, t) + 0.1 * bce_loss(p, t)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_dice = 0.0
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # 7) Boucle d’entraînement
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # ─── Entraînement ──────────────────────────────────────────
        model.train()
        train_loss = train_dice = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{MAX_EPOCHS} [train]")
        for step, (imgs, masks) in enumerate(pbar, 1):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            with autocast():
                preds = model(imgs)
                l     = loss_fn(preds, masks) / ACCUM_STEPS

            scaler.scale(l).backward()
            if step % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += l.item() * ACCUM_STEPS
            train_dice += dice(torch.sigmoid(preds), masks).item()
            pbar.set_postfix(loss=f"{train_loss/step:.4f}", dice=f"{train_dice/step:.4f}")

        scheduler.step()

        # ─── Validation ────────────────────────────────────────────
        model.eval()
        val_loss = val_dice = val_iou = 0.0

        with torch.no_grad(), autocast():
            for imgs, masks in tqdm(dl_val, desc="val", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)

                val_loss += loss_fn(preds, masks).item()
                val_dice += dice(torch.sigmoid(preds), masks).item()
                val_iou  += iou(torch.sigmoid(preds), masks).item()

        # libère le cache GPU pour éviter fragmentation
        torch.cuda.empty_cache()

        # moyennes
        n_tr = len(dl_train)
        n_va = len(dl_val)
        avg_tr_loss = train_loss / n_tr
        avg_tr_dice = train_dice / n_tr
        avg_va_loss = val_loss / n_va
        avg_va_dice = val_dice / n_va
        avg_va_iou  = val_iou / n_va

        print(
            f"🚀 Epoch {epoch:02d} | "
            f"train_loss {avg_tr_loss:.4f} / dice {avg_tr_dice:.4f} | "
            f"val_loss {avg_va_loss:.4f} / dice {avg_va_dice:.4f} / iou {avg_va_iou:.4f} | "
            f"{(time.time() - t0)/60:.1f} min"
        )

        # ─── Checkpoints ───────────────────────────────────────────
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "dice": avg_va_dice},
            f"{OUTPUT_DIR}/last.pth"
        )
        if avg_va_dice > best_dice:
            best_dice = avg_va_dice
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_dice.pth")
            print(f"✅ Nouveau meilleur Dice : {best_dice:.4f}")

    print(f"🏁 Entraînement terminé — meilleur Dice val = {best_dice:.4f}")

# ─── Windows : safe spawn ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
