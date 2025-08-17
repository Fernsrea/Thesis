import os, glob, random
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import albumentations as A
import albumentations.pytorch as AT
import pytorch_lightning as pl

def load_img(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.array(img)

def load_mask(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    m = (np.array(m) > 127).astype(np.uint8)
    return m

class ISICInpaintDataset(Dataset):
    def __init__(self, root, size=256, mask_suffix="_segmentation.png",
                 random_irregular_prob=0.3, preserve_lesion_mask=True):
        self.root = root
        self.size = size
        self.random_irregular_prob = random_irregular_prob
        self.preserve_lesion_mask = preserve_lesion_mask
        img_dir = os.path.join(root, "images")
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + \
                     sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_dir = os.path.join(root, "masks")
        self.mask_suffix = mask_suffix

        self.to_tensor = AT.ToTensorV2()

        # simple geometric/color augs
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(0.05, 0.05, 0.05, 0.02, p=0.3),
            AT.ToTensorV2()
        ])

    def _mask_path(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.mask_dir, f"{stem}{self.mask_suffix}")

    def _random_irregular_mask(self, h, w):
        # quick irregular mask: random strokes
        m = np.zeros((h, w), np.uint8)
        num_strokes = np.random.randint(1, 6)
        for _ in range(num_strokes):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            thickness = np.random.randint(10, 40)
            cv2 = __import__("cv2")
            cv2.line(m, (x1, y1), (x2, y2), 1, thickness=thickness)
        return m

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        ip = self.paths[idx]
        mp = self._mask_path(ip)
        img = load_img(ip, self.size)
        if os.path.exists(mp):
            mask = load_mask(mp, self.size)
        else:
            mask = np.zeros((self.size, self.size), np.uint8)

        # mix in irregular mask for robustness
        if np.random.rand() < self.random_irregular_prob:
            irr = self._random_irregular_mask(self.size, self.size)
            if self.preserve_lesion_mask:
                mask = np.clip(mask | irr, 0, 1)
            else:
                mask = irr

        # apply augs and to tensor
        out = self.aug(image=img, mask=mask.astype(np.uint8)*255)
        x = out["image"].float() / 255.0  # [0,1]
        m = (out["mask"] > 127).float()   # {0,1}

        # normalize to [-1,1]
        x = x * 2 - 1.0
        x_masked = x * (1.0 - m)

        return {"image": x, "mask": m, "masked": x_masked}

class ISICInpaintDataModule(pl.LightningDataModule):
    def __init__(self, root, size=256, batch_size=8, num_workers=4, val_frac=0.1, seed=42):
        super().__init__()
        self.root, self.size = root, size
        self.batch_size, self.num_workers = batch_size, num_workers
        self.val_frac, self.seed = val_frac, seed

    def setup(self, stage: Optional[str] = None):
        full = ISICInpaintDataset(self.root, size=self.size)
        n = len(full); nv = int(self.val_frac * n); nt = n - nv
        gen = torch.Generator().manual_seed(self.seed)
        self.train_set, self.val_set = random_split(full, [nt, nv], generator=gen)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False)
