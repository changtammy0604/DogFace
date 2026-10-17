import torch
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform
from torch.utils.data import DataLoader

# =======================================
# 1. 參數設定
# =======================================
# 設定影像大小，SimCLRTrainDataTransform()會經由 RandomResizedCrop 將圖片統一裁切到設定的大小
input_height = 224
train_path = "train"
valid_path = "valid"

# NVIDIA GeForce RTX 3090 可以犧牲一點精度來換效能
torch.set_float32_matmul_precision('high')

# =======================================
# 2. 載入訓練資料集
# ========================================
# SimCLRTrainDataTransform 會回傳一個 tuple: (img_i, img_j, label)
train_dataset = ImageFolder(
    root=train_path,
    transform=SimCLRTrainDataTransform(input_height=input_height)
)

valid_dataset = ImageFolder(
    root=valid_path,
    transform=SimCLREvalDataTransform(input_height=input_height)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=8)
train_samples = len(train_loader)
train_samples = len(train_loader)

# ========================================
# 3. 初始化 SimCLR 模型
# ========================================
model = SimCLR(
    gpus=1,
    num_samples=train_samples,
    batch_size=256,
    dataset='Sibetan',
)

print("模型已初始化")
print(train_loader.batch_size)