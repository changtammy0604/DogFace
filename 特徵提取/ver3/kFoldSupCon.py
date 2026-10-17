#!/usr/bin/env python
# coding: utf-8
# ==========================================================
# 用於訓練"純SupCon"，已加上 k-fold 和 mAP
# ==========================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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

# 訓練集&驗證集路徑
dataset_path = "multi_body/train_valid_all/"

# 固定模型權重初始化、MPerClassSampler、StratifiedKFold的隨機種子，以便實驗的可重複性
pl.seed_everything(26, workers=True)

# ==========================================
# 2. 定義雙視圖資料增強
# ==========================================
class DualViewTransform:
    # 將一張圖片轉換為兩個隨機增強版本
    def __init__(self, transform):
      self.transform = transform

    def __call__(self, x):
      return self.transform(x), self.transform(x)

# 訓練期增強：保留裁切與翻轉，但降低顏色破壞，保留狗狗毛色
train_transform = transforms.Compose([
  transforms.RandomResizedCrop(input_height, scale=(0.4, 1.0)), # 裁切範圍調大，確保包含狗主體
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 輕微顏色擾動
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 驗證期增強：穩定裁切
valid_transform = transforms.Compose([
  transforms.Resize(int(input_height * 1.14)),
  transforms.CenterCrop(input_height),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 定義 SupCon Lightning 模型
# ==========================================
class SupConModel(pl.LightningModule):
  def __init__(self, backbone='resnet50', projection_dim=128, embed_dim=2048, temperature=0.1):
    super().__init__()
    self.save_hyperparameters()
    self.gallery_loader = None

    # 1. 載入預訓練 ResNet50 並拔除最後一層
    self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    self.n_features = self.encoder.fc.in_features
    self.encoder.fc = nn.Identity()

    # 2. 降維層：推論/檢索只用到這裡，維度由 embed_dim 控制
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

  def extract_feature(self, x):
    # 做檢索比對時用這個：只回傳降維後的特徵，不經過 MLP，並強制 eval 模式。
    was_training = self.training
    self.eval()
    with torch.no_grad():
      h = self.encoder(x)
      r = self.reduction(h)
    if was_training:
      self.train()
    return r

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

  def on_validation_epoch_start(self):
    self._val_embeds, self._val_labels = [], []

  def validation_step(self, batch, batch_idx):
    images, labels = batch
    emb = self.extract_feature(images)
    self._val_embeds.append(emb.float().cpu())
    self._val_labels.append(labels.cpu())
    
  def on_validation_epoch_end(self):
    query_emb = torch.cat(self._val_embeds, dim=0)
    query_lbl = torch.cat(self._val_labels, dim=0)

    gallery_embeds, gallery_labels = [], []
    with torch.no_grad():
      for imgs, lbls in self.gallery_loader:
        imgs = imgs.to(self.device)
        gallery_embeds.append(self.extract_feature(imgs).float().cpu())
        gallery_labels.append(lbls)
    gallery_emb = torch.cat(gallery_embeds, dim=0)
    gallery_lbl = torch.cat(gallery_labels, dim=0)

    q = F.normalize(query_emb, dim=1)
    g = F.normalize(gallery_emb, dim=1)
    sim = q @ g.T                       # [num_query, num_gallery]

    # ---- rank-1 ----
    top1_idx = sim.argmax(dim=1)
    top1_pred = gallery_lbl[top1_idx]
    rank1_acc = (top1_pred == query_lbl).float().mean()
    self.log("val_rank1", rank1_acc, prog_bar=True)

    # ---- mAP ----
    APs = []
    for i in range(sim.size(0)):
      order = torch.argsort(sim[i], descending=True)
      matches = (gallery_lbl[order] == query_lbl[i]).float()
      num_relevant = matches.sum()
      if num_relevant == 0:
        continue   # 這隻狗在 gallery 裡沒有其他照片，這個 query 無法評估，跳過
      cum_matches = torch.cumsum(matches, dim=0)
      ranks = torch.arange(1, len(matches) + 1, dtype=torch.float)
      precision_at_k = cum_matches / ranks
      AP = (precision_at_k * matches).sum() / num_relevant
      APs.append(AP.item())

    mAP = sum(APs) / len(APs) if APs else 0.0
    self.log("val_mAP", mAP, prog_bar=True)

    coverage = len(APs) / sim.size(0)
    self.log("val_mAP_coverage", coverage, prog_bar=True)
    
    self.log("val_gallery_size", float(gallery_emb.size(0)), prog_bar=False)
    
  def configure_optimizers(self):
    # 剛開始使用較小的 Learning Rate
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return [optimizer], [scheduler]

# ==========================================
# 4. 準備資料集與 Sampler
# ==========================================
dataset = ImageFolder(dataset_path, transform=None)  # 等分完 fold 再做 transform
labels = np.array([label for _, label in dataset.samples])

# 處裡 valid 和 train 需要不同 transform 的問題
class TransformSubset(Dataset):
  def __init__(self, base, indices, transform):
    self.base, self.indices, self.transform = base, indices, transform
  def __len__(self):
    return len(self.indices)
  def __getitem__(self, i):
    img, label = self.base[self.indices[i]]
    if self.transform:
      img = self.transform(img)
    return img, label

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
  train_dataset = TransformSubset(dataset, train_idx, DualViewTransform(train_transform))
  valid_dataset = TransformSubset(dataset, val_idx, valid_transform)         # 單視圖，當作用戶上傳的圖片
  gallery_dataset = TransformSubset(dataset, train_idx, valid_transform)     # 單視圖、無增強，作為 gallery <= 這裡用train作為gallery(資料庫裡的圖片)，

  # 使用 MPerClassSampler，確保每個 Batch 裡每隻狗都有 M 張照片來做正樣本配對
  # 假設 m=4，代表每個 batch 中每隻抽到的狗都會有 4 張照片
  sampler = MPerClassSampler(
    labels=labels[train_idx],
    m=4,
    batch_size=batch_size,
    length_before_new_iter=len(train_dataset)
  )
  
  # 使用了 sampler 就不可以設定 shuffle=True
  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
  gallery_loader = DataLoader(gallery_dataset, batch_size=256, shuffle=False, num_workers=8, drop_last=False)
  
# ==========================================
# 5. 開始測試
# ==========================================
  print("===============Initialize Model===============")
  model = SupConModel()
  model.gallery_loader = gallery_loader
  
  logger = TensorBoardLogger(save_dir=f"lightning_logs/SupCon_{dataset_path.split('/')[0]}/", name=f"fold{k}")
  
  # 將 valid 結果最好的那組參數保留下來
  checkpoint_cb = ModelCheckpoint(
    monitor="val_mAP",
    mode="max",
    save_top_k=1,
    dirpath=f"checkpoints/fold{k}",
    filename="best-{epoch}-{val_mAP:.4f}-{val_rank1:.4f}"
  )
  
  trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    precision=16,            # 16-bit 混和精度省空間
    max_epochs=max_epochs,
    log_every_n_steps=10,    # 避免 log 印太快，改為 10 步印一次
    callbacks=[checkpoint_cb],
    enable_progress_bar=False,
    logger=logger
  )
  
  print("===============Current Parameter===============")
  print(f"batch_size = {batch_size}")
  print(f"max_epochs = {max_epochs}")
  
  trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
  print(f"Best model saved in: {checkpoint_cb.best_model_path}")