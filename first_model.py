# connected_unet_finetune_from_best.py
# Continuer l’entraînement U-Net++ (ResNet-50) à partir du meilleur checkpoint (epoch 40, Dice = 0.4992)
# avec patch‐cropping centré sur la lésion et optimisations GPU (RTX 4060)

import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import TverskyLoss

# ───────── CONFIGURATION ─────────
DATA_ROOT   = Path(r"C:\Users\PC\Desktop\DATASET_CLEAN")
IMAGE_ROOT  = DATA_ROOT / "images"
MASK_ROOT   = DATA_ROOT / "masks"
SAVE_DIR    = Path(r"C:\Users\PC\Desktop\models_connected_gpu")
SAVE_DIR.mkdir(exist_ok=True)

# Chemin exact vers votre checkpoint epoch 40 (Dice 0.4992)
CHECKPOINT_PATH = SAVE_DIR / "unetpp_gpu_ep40_dice0.4992.pth"

PATCH_SIZE   = 160    # Taille du patch carré extrait autour de la lésion
IMG_SIZE     = 256    # Taille d’entrée du modèle après resize Albumentations
BATCH_SIZE   = 8      # Réduire si OOM, remonter si VRAM dispo
NUM_EPOCHS   = 40     # Nouveaux epochs de finetune
LR_MAX       = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.20   # Fraction par patient pour validation
RANDOM_SEED  = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)

print("🚀 Device :", DEVICE)

USE_AMP     = (DEVICE.type == "cuda")
NUM_WORKERS = max(1, os.cpu_count() // 2)


# ───────── extract_patch ─────────
def extract_patch(img: np.ndarray, msk: np.ndarray, patch_size: int = 160):
    """
    img : numpy array H×W×3 (RGB)
    msk : numpy array H×W (valeurs 0 ou >0)
    patch_size : taille du patch carré à extraire (en pixels)
    Retourne (img_patch, msk_patch) de dimensions patch_size×patch_size
    """
    ys, xs = np.where(msk > 0)
    h, w = img.shape[:2]
    if len(xs) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(np.mean(ys)), int(np.mean(xs))

    half = patch_size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = y0 + patch_size
    x1 = x0 + patch_size

    # Ajuster si on dépasse les bords
    if y1 > h:
        y1 = h
        y0 = h - patch_size
    if x1 > w:
        x1 = w
        x0 = w - patch_size
    y0, x0 = max(0, y0), max(0, x0)

    img_patch = img[y0:y0+patch_size, x0:x0+patch_size]
    msk_patch = msk[y0:y0+patch_size, x0:x0+patch_size]
    return img_patch, msk_patch


# ───────── DATASET AVEC PATCH‐CROPPING ─────────
class CMMDatasetPatch(Dataset):
    """
    Pour chaque image :
      1) Charger image complète (RGB) + masque complet (L)
      2) Extraire un patch carré de taille PATCH_SIZE centré sur la lésion
      3) Appliquer Albumentations (resize PATCH→IMG_SIZE, augmentations, normalisation)
      4) Retourner (img_tensor, mask_tensor)
    """
    def __init__(self, img_paths, transform=None, patch_size: int = 160):
        self.img_paths   = img_paths
        self.mask_paths  = [
            (MASK_ROOT / p.relative_to(IMAGE_ROOT).parent / f"{p.stem}_mask.png")
            for p in img_paths
        ]
        self.transform   = transform
        self.patch_size  = patch_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        msk_path = self.mask_paths[idx]

        # 1) Charger full image et full mask en numpy
        img_np = np.array(Image.open(img_path).convert("RGB"))
        msk_np = np.array(Image.open(msk_path).convert("L"))

        # 2) Extraire patch centré sur la lésion
        img_patch, msk_patch = extract_patch(img_np, msk_np, patch_size=self.patch_size)

        # 3) Appliquer Albumentations sur le patch (resize à IMG_SIZE, augmentations)
        if self.transform:
            aug = self.transform(image=img_patch, mask=msk_patch)
            x   = aug["image"]
            y   = (aug["mask"] > 0).float().unsqueeze(0)
        else:
            x = torch.from_numpy(img_patch.transpose(2, 0, 1) / 255.0).float()
            y = torch.from_numpy((msk_patch > 0).astype("f")).unsqueeze(0)

        return x, y


# ───────── SPLIT PAR PATIENT ─────────
def split_by_patient(val_frac=0.2, seed=42):
    random.seed(seed)
    patients = {}
    for p in IMAGE_ROOT.rglob("*.png"):
        pid = p.relative_to(IMAGE_ROOT).parts[0]
        patients.setdefault(pid, []).append(p)
    ids = list(patients)
    random.shuffle(ids)
    n_val   = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train, val = [], []
    for pid, lst in patients.items():
        if pid in val_ids:
            val.extend(lst)
        else:
            train.extend(lst)
    return train, val


# ───────── TRANSFORMS AVEC CLAHE + GAMMA ─────────
def get_transforms():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(var_limit=(10,50), p=0.4),
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


# ───────── LOSS & MÉTRIQUE ─────────
loss_fn = TverskyLoss(mode="binary", alpha=0.7, beta=0.3)

@torch.no_grad()
def dice_score(pred, tgt, thr=0.5):
    prob = torch.sigmoid(pred)
    mask = (prob > thr).float()
    inter = (mask * tgt).sum([1,2,3])
    union = mask.sum([1,2,3]) + tgt.sum([1,2,3])
    return ((2 * inter + 1e-7) / (union + 1e-7)).mean().item()


# ───────── FONCTION D’ENTRAÎNEMENT / VALIDATION ─────────
def run_epoch(model, loader, optimizer=None, scaler=None, scheduler=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0

    for imgs_cpu, masks_cpu in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
        imgs  = imgs_cpu.to(DEVICE, non_blocking=True)
        masks = masks_cpu.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            preds = model(imgs)
            loss  = loss_fn(preds, masks)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        dice = dice_score(preds, masks)
        bs   = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += dice * bs

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


# ───────── MAIN ─────────
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)

    # 1) Split 80%/20% PAR PATIENT
    train_paths, val_paths = split_by_patient(VAL_FRAC, RANDOM_SEED)
    print(f"• Images d'entraînement : {len(train_paths)}")
    print(f"• Images de validation  : {len(val_paths)}")

    # 2) DataLoaders sur PATCH
    train_tf, val_tf = get_transforms()
    train_ds = CMMDatasetPatch(train_paths, transform=train_tf, patch_size=PATCH_SIZE)
    val_ds   = CMMDatasetPatch(val_paths,   transform=val_tf,   patch_size=PATCH_SIZE)

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

    # 3) Instancier U-Net++ (ResNet-50) et charger le best checkpoint ep40
    model = smp.UnetPlusPlus(
        encoder_name    = "resnet50",
        encoder_weights = None,   # on ne recharge pas ImageNet, on va charger le checkpoint complet
        in_channels     = 3,
        classes         = 1
    ).to(DEVICE)

    print("🔄 Chargement du checkpoint ep40 :", CHECKPOINT_PATH.name)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    # 4) Optimizer, Scheduler, Scaler
    optimizer = AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    steps     = len(train_loader) * NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr        = LR_MAX,
        total_steps   = steps,
        pct_start     = 0.1,
        anneal_strategy="cos"
    )
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # 5) Finetune 40 epochs de plus
    best_val_dice = 0.4992   # valeur du Dice à l’epoch 40 précédent
    history = {"tr_loss": [], "tr_dice": [], "vl_loss": [], "vl_dice": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Finetune Epoch {epoch}/{NUM_EPOCHS} =====")
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer, scaler, scheduler)
        vl_loss, vl_dice = run_epoch(model, val_loader)

        print(f"▶ Train — Loss: {tr_loss:.4f} | Dice: {tr_dice:.4f}")
        print(f"◆ Val   — Loss: {vl_loss:.4f} | Dice: {vl_dice:.4f}")

        history["tr_loss"].append(tr_loss)
        history["tr_dice"].append(tr_dice)
        history["vl_loss"].append(vl_loss)
        history["vl_dice"].append(vl_dice)

        # Sauvegarde du nouveau meilleur modèle si Dice validation s'améliore
        if vl_dice > best_val_dice:
            best_val_dice = vl_dice
            ckpt_name = f"finetune_unetpp_ep{epoch:02d}_dice{vl_dice:.4f}.pth"
            ckpt_path = SAVE_DIR / ckpt_name
            torch.save(model.state_dict(), ckpt_path)
            print("💾 Nouveau meilleur checkpoint finetune :", ckpt_name)

    # 6) Affichage des courbes Loss & Dice (fine-tuning)
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["tr_loss"], label="Train Loss")
    plt.plot(epochs, history["vl_loss"], label="Val   Loss")
    plt.title("Loss (Tversky) — Finetune")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["tr_dice"], label="Train Dice")
    plt.plot(epochs, history["vl_dice"], label="Val   Dice")
    plt.title("Dice Score — Finetune")
    plt.xlabel("Epoch"); plt.ylabel("Dice")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()
