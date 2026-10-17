#!/usr/bin/env python
# coding: utf-8
# ==========================================================
# 用於訓練"純SupCon"
# ==========================================================

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 導入 SupCon 必備套件
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler

# ==========================================
# 1. 參數設定
# ==========================================
batch_size = 128 
max_epochs = 400
input_height = 224

# NVIDIA GeForce RTX 3090 可以犧牲一點精度來換效能
torch.set_float32_matmul_precision('high')

# ==========================================
# 2. 個體辨識的雙視圖資料增強
# ==========================================
class DualViewTransform:
    # 將一張圖片轉換為兩個隨機增強版本
    def __init__(self, transform):
      self.transform = transform

    def __call__(self, x):
      return self.transform(x), self.transform(x)

# 訓練期增強：保留裁切與翻轉，但降低顏色破壞，保留狗狗毛色
train_base_transform = transforms.Compose([
  transforms.RandomResizedCrop(input_height, scale=(0.4, 1.0)), # 裁切範圍調大，確保包含狗主體
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 輕微顏色擾動
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 驗證期增強：穩定裁切
valid_base_transform = transforms.Compose([
  transforms.Resize(int(input_height * 1.14)),
  transforms.CenterCrop(input_height),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 準備資料集與 Sampler
# ==========================================
train_dataset = ImageFolder(
  root="/home/dogfaceid/muti_face/train/",
  transform=DualViewTransform(train_base_transform) # 回傳 ((img_i, img_j), label)
)

valid_dataset = ImageFolder(
  root="/home/dogfaceid/muti_face/valid/",
  transform=DualViewTransform(valid_base_transform)
)

# 使用 MPerClassSampler，確保每個 Batch 裡每隻狗都有 M 張照片來做正樣本配對
# 假設 m=4，代表每個 batch 中每隻抽到的狗都會有 4 張照片
sampler = MPerClassSampler(
  labels=train_dataset.targets,
  m=4,
  batch_size=batch_size,
  length_before_new_iter=len(train_dataset)
)

# 使用了 sampler 就不可以設定 shuffle=True
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

# ==========================================
# 4. 定義 SupCon Lightning 模型
# ==========================================
class SupConModel(pl.LightningModule):
  def __init__(self, backbone='resnet50', projection_dim=128, embed_dim=2048, temperature=0.1):
      super().__init__()
      self.save_hyperparameters()

      # 1. 載入預訓練 ResNet50 並拔除最後一層
      self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
      self.n_features = self.encoder.fc.in_features
      self.encoder.fc = nn.Identity()

      # 2. 降維層：推論與檢索只用到這裡，維度由 embed_dim 控制
      if embed_dim != self.n_features:
          self.reduction = nn.Linear(self.n_features, embed_dim)
      else:
          # 如果維度相同，則減少不必要的參數與運算
          self.reduction = nn.Identity()

      # 3. 建立 Projection Head (MLP)
      self.projector = nn.Sequential(
          nn.Linear(embed_dim, embed_dim),
          nn.BatchNorm1d(embed_dim),
          nn.ReLU(),
          nn.Linear(embed_dim, projection_dim)
      )
      
      # 4. 宣告 SupCon Loss
      self.criterion = SupConLoss(temperature=temperature)

  def forward(self, x):
    # 推理時呼叫 forward，保留 projector 以備不時之需，
    # 但記得實際配對時，只需取用 self.encoder(x) 即可。
    h = self.encoder(x)
    r = self.reduction(h)
    z = self.projector(r)
    return r, z

  def shared_step(self, batch, mode="train"):
    (x_i, x_j), labels = batch
    
    # 兩個視圖疊進同一個 batch，一起 forward
    x_cat = torch.cat([x_i, x_j], dim=0)
    cat_labels = torch.cat([labels, labels], dim=0)

    _, z = self(x_cat)

    # SupCon Loss 會自動根據 cat_labels 把同 ID 的特徵拉近，不同 ID 的推開
    loss = self.criterion(z, cat_labels)
    
    self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss

  def training_step(self, batch, batch_idx):
    return self.shared_step(batch, mode="train")

  def validation_step(self, batch, batch_idx):
    return self.shared_step(batch, mode="val")

  def configure_optimizers(self):
    # 剛開始使用較小的 Learning Rate
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return [optimizer], [scheduler]

  def extract_feature(self, x):
    # 做檢索比對時用這個，只回傳降維後的特徵，不經過 MLP，並強制 eval 模式。
    was_training = self.training
    self.eval()
    with torch.no_grad():
      h = self.encoder(x)
      r = self.reduction(h)
    if was_training:
      self.train()
    return r

# ==========================================
# 5. 開始訓練
# ==========================================
print("=============== 初始化模型 ===============")
model = SupConModel()

trainer = pl.Trainer(
  accelerator='gpu',
  devices=1,
  precision=16,           # 16-bit 混和精度省空間
  max_epochs=max_epochs,
  log_every_n_steps=10    # 避免 log 印太快，改為 10 步印一次
)

print("=============== 目前參數 ===============")
print(f"batch_size = {batch_size}")
print(f"max_epochs = {max_epochs}")

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)