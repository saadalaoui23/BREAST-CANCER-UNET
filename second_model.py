# connected_unet_quick_eval.py
# Entraînement rapide sur 10 % du dataset avec U-Net++ (ResNet50)
# Affiche Dice et Loss pour chaque epoch

import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import TverskyLoss

# ───────── CONFIGURATION ─────────
DATA_ROOT    = Path(r"C:\Users\PC\Desktop\DATASET_CLEAN")
IMAGE_ROOT   = DATA_ROOT / "images"
MASK_ROOT    = DATA_ROOT / "masks"
SAVE_DIR     = Path(r"C:\Users\PC\Desktop\models_quick"); SAVE_DIR.mkdir(exist_ok=True)

IMG_SIZE      = 256
BATCH_SIZE    = 16
NUM_EPOCHS    = 10                # Entraînement rapide sur 10 epochs
LR_MAX        = 1e-3
WEIGHT_DECAY  = 1e-4
SAMPLE_FRAC   = 0.10              # Utiliser 10 % des images pour train + val
VAL_FRAC      = 0.20              # 20 % de ce sous-ensemble pour validation
RANDOM_SEED   = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    # Réserver 90 % de la VRAM
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)
print("🚀 Device :", DEVICE)
USE_AMP      = (DEVICE.type == "cuda")
NUM_WORKERS  = max(1, os.cpu_count() // 2)

# ───────── DATASET & TRANSFORMS ─────────
class CMMDataset(Dataset):
    def __init__(self, img_paths, tf=None):
        self.img_paths = img_paths
        self.mask_paths = []
        for p in img_paths:
            rel = p.relative_to(IMAGE_ROOT)
            mask_folder = MASK_ROOT / rel.parent
            mask_name = rel.stem + "_mask.png"
            self.mask_paths.append(mask_folder / mask_name)
        self.tf = tf

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(mask_path).convert("L"))
        if self.tf:
            aug = self.tf(image=img, mask=msk)
            x = aug["image"]
            y = (aug["mask"] > 0).float().unsqueeze(0)
        else:
            x = torch.from_numpy(img.transpose(2,0,1) / 255.0).float()
            y = torch.from_numpy((msk > 0).astype(np.float32)).unsqueeze(0)
        return x, y

def get_transforms():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(var_limit_range=(10,50), p=0.4),
        A.GaussianBlur(blur_limit=(3,7), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ], additional_targets={"mask":"mask"})

    val_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean, std),
        ToTensorV2(),
    ], additional_targets={"mask":"mask"})
    return train_tf, val_tf

# ───────── SPLIT 10 % DU DATASET ─────────
all_img_paths = list(IMAGE_ROOT.rglob("*.png"))
random.seed(RANDOM_SEED)
random.shuffle(all_img_paths)

n_sample = max(1, int(len(all_img_paths) * SAMPLE_FRAC))
sample_paths = all_img_paths[:n_sample]

n_val = max(1, int(len(sample_paths) * VAL_FRAC))
train_paths = sample_paths[n_val:]
val_paths   = sample_paths[:n_val]

print(f"• Taille sous-ensemble total : {len(sample_paths)} images")
print(f"• Entraînement (80 %) : {len(train_paths)} images")
print(f"• Validation   (20 %) : {len(val_paths)} images")

# ───────── CONSTRUCT DATALOADERS ─────────
train_tf, val_tf = get_transforms()
train_ds = CMMDataset(train_paths, tf=train_tf)
val_ds   = CMMDataset(val_paths,   tf=val_tf)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# ───────── MODÈLE U-Net++ ─────────
model = smp.UnetPlusPlus(
    encoder_name    = "resnet50",
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = 1,
).to(DEVICE)

# ───────── LOSS & MÉTRIQUE ─────────
loss_fn = TverskyLoss(mode="binary", alpha=0.7, beta=0.3)

@torch.no_grad()
def dice_metric(pred, tgt, thr=0.5):
    prob = torch.sigmoid(pred)
    mask = (prob > thr).float()
    B = tgt.size(0)
    inter = (mask * tgt).view(B, -1).sum(dim=1)
    union = mask.view(B, -1).sum(dim=1) + tgt.view(B, -1).sum(dim=1)
    dice = ((2*inter + 1e-7) / (union + 1e-7)).mean().item()
    return dice

# ───────── TRAIN / VALIDATION FUNCTION ─────────
def run_epoch(model, loader, optimizer=None, scaler=None, scheduler=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    for imgs, masks in tqdm(loader, desc=("Train" if is_train else "Val"), leave=False):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(imgs)
            loss = loss_fn(preds, masks)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        d = dice_metric(preds, masks)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += d * bs

    n = len(loader.dataset)
    return total_loss / n, total_dice / n

# ───────── OPTIMIZER & SCHEDULER ─────────
optimizer = AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
steps = len(train_loader) * NUM_EPOCHS
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR_MAX,
    total_steps=steps,
    pct_start=0.1,
    anneal_strategy="cos"
)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ───────── BOUCLE D’ENTRAÎNEMENT RAPIDE ─────────
best_val_dice = 0.0
history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
    tr_loss, tr_dice = run_epoch(model, train_loader, optimizer, scaler, scheduler)
    vl_loss, vl_dice = run_epoch(model, val_loader)

    print(f"▶ Train — Loss: {tr_loss:.4f} | Dice: {tr_dice:.4f}")
    print(f"◆ Val   — Loss: {vl_loss:.4f} | Dice: {vl_dice:.4f}")

    history["train_loss"].append(tr_loss)
    history["train_dice"].append(tr_dice)
    history["val_loss"].append(vl_loss)
    history["val_dice"].append(vl_dice)

    if vl_dice > best_val_dice:
        best_val_dice = vl_dice
        ckpt_path = SAVE_DIR / f"unetpp_ep{epoch:02d}_dice{vl_dice:.4f}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Nouveau meilleur modèle sauvegardé : {ckpt_path.name} (Dice={vl_dice:.4f})")

# ───────── PLOT DES CURVES ─────────
epochs = np.arange(1, NUM_EPOCHS + 1)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"],   label="Val   Loss")
plt.title("Loss (Tversky)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, history["train_dice"], label="Train Dice")
plt.plot(epochs, history["val_dice"],   label="Val   Dice")
plt.title("Dice Score")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()
