import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import math
import copy

from DIPnet import DIPNet                    # <- Import mô hình DIPNet gốc của bạn
from quantize import quantize_model, freeze_model, unfreeze_model
from quantize import TrainWholeDataset, ValDataset  # <- Đảm bảo bạn có các class Dataset

# ==== CONFIG ====
MODEL_PATH = "./pretrained/dipnet.pth"       # Trọng số DIPNet trước QAT
SAVE_PATH = "./dipnet_qat_final.pth"         # File sẽ lưu model sau QAT
TRAIN_DIR = "./data/DIV2K_train_HR"
VAL_DIR = "./data/DIV2K_valid_HR"
CROP_SIZE = 48
UPSCALE = 4
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BITS = 8  # Số bit lượng hóa cho cả weight và activation
# ==============

def load_pretrained_model():
    model = DIPNet()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    return model

def get_dataloaders():
    train_set = TrainWholeDataset([TRAIN_DIR], crop_size=CROP_SIZE, upscale_factor=UPSCALE)
    val_set = ValDataset([VAL_DIR], crop_size=CROP_SIZE, upscale_factor=UPSCALE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader

def psnr(sr, hr):
    mse = ((sr - hr) ** 2).mean()
    return 10 * math.log10(1.0 / mse.item())

def train_qat(model, train_loader, val_loader):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === Phase 1: Warm-up để cập nhật thống kê min/max ===
    print("🔄 Phase 1: Warm-up - updating activation stats")
    unfreeze_model(model)
    model.train()
    for epoch in range(1):
        loop = tqdm(train_loader, desc=f"Warm-up Epoch {epoch+1}")
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # === Phase 2: QAT Fine-tune với freeze activation stats ===
    print("🎯 Phase 2: QAT Fine-tuning")
    freeze_model(model)
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}")
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        total_psnr = 0
        with torch.no_grad():
            for val_lr, val_hr in val_loader:
                val_lr = val_lr.to(DEVICE)
                val_hr = val_hr.to(DEVICE)
                sr = model(val_lr)
                total_psnr += psnr(sr, val_hr)
        avg_psnr = total_psnr / len(val_loader)
        print(f"📊 Validation PSNR: {avg_psnr:.2f} dB")

    # Save model
    torch.save({"model": model.state_dict()}, SAVE_PATH)
    print(f"✅ Saved quantized model to {SAVE_PATH}")

if __name__ == "__main__":
    print("🚀 Starting QAT for DIPNet...")
    model_fp32 = load_pretrained_model().to(DEVICE)
    model_qat = quantize_model(model_fp32, weight_bit=NUM_BITS, act_bit=NUM_BITS, full_precision_flag=False).to(DEVICE)
    train_loader, val_loader = get_dataloaders()
    train_qat(model_qat, train_loader, val_loader)
