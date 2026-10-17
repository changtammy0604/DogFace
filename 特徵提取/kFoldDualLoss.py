#!/usr/bin/env python
# coding: utf-8

# ==========================================================
# 用於訓練"SupCon+CrossEntropy"
# ==========================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 載入 SupCon 必備套件
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler

# ==========================================
# 1. 參數設定
# ==========================================
batch_size = 128
max_epochs = 200
input_height = 224
output_dim = 2048
part = "face"
warmup_epochs = 10

# NVIDIA GeForce RTX 3090 可以犧牲一點精度來換效能
torch.set_float32_matmul_precision('high')

# 訓練集&驗證集路徑
dataset_path = "dogfaceid_face/train_valid_all"

# 固定模型權重初始化、MPerClassSampler、StratifiedKFold的隨機種子，以便實驗的可重複性
pl.seed_everything(26, workers=True)

# ==========================================
# 2. 定義雙視圖數據增強
# ==========================================
class DualViewTransform:
    # 將一張圖片轉換為兩個隨機增強版本
    def __init__(self, transform):
      self.transform = transform

    def __call__(self, x):
      return self.transform(x), self.transform(x)

# 訓練期增強：保留裁切與翻轉，但降低顏色破壞，保留毛色
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_height, scale=(0.4, 1.0)), # 裁切範圍調大，確保包含狗主體
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 輕微顏色擾動
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 驗證期增強：僅套用裁切
valid_transform = transforms.Compose([
    transforms.Resize(int(input_height * 1.14)),
    transforms.CenterCrop(input_height),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 定義 Dual Loss + BNNeck 模型
# ==========================================
class DualLossModel(pl.LightningModule):
  def __init__(self, num_classes, backbone='resnet50', feat_dim=2048, temperature=0.1):
    super().__init__()
    self.save_hyperparameters()
    self.gallery_loader = None

    # 1. 載入預訓練 ResNet50 並拔除原始分類層
    self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    self.backbone_dim = self.encoder.fc.in_features
    self.encoder.fc = nn.Identity()
    
    # 2. 如果指定的 feat_dim 與 Backbone 輸出不同，則加入 Linear 層降維/升維
    if feat_dim != self.backbone_dim:
      self.bottleneck = nn.Linear(self.backbone_dim, feat_dim, bias=False)
      nn.init.kaiming_normal_(self.bottleneck.weight, a=0, mode='fan_out')
    else:
      # 如果維度相同，則減少不必要的參數與運算
      self.bottleneck = nn.Identity()

    # 3. BNNeck：將特徵正規化，解決 Metric Loss 與 ID Loss 的梯度衝突
    self.bnneck = nn.BatchNorm1d(feat_dim)
    self.bnneck.bias.requires_grad_(False)  # 凍結 bias 提升收斂穩定性 (參考 BoT)
    
    # 4. Classifier：用於計算 ID Loss
    self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
    
    # 5. 初始化 BN 與 Classifier 權重
    nn.init.normal_(self.classifier.weight, std=0.001)
    nn.init.constant_(self.bnneck.weight, 1.0)
    nn.init.constant_(self.bnneck.bias, 0.0)

    # 6. 宣告雙重 Loss
    self.criterion_supcon = SupConLoss(temperature=temperature)
    self.criterion_id = nn.CrossEntropyLoss(label_smoothing=0.1)

  def forward(self, x):
    f_raw = self.encoder(x)                # backbone 原始特徵 (2048 維)
    f_before_bn = self.bottleneck(f_raw)    # 降維/升維後、尚未過 BN，給 SupCon 用
    f_after_bn = self.bnneck(f_before_bn)    # 過 BN 後，給 ID Loss 用
    logits = self.classifier(f_after_bn)
    return f_before_bn, f_after_bn, logits
    
  def extract_feature(self, x):
    # 做檢索比對時用這個：只回傳 BN 後的特徵，並強制 eval 模式。
    was_training = self.training
    self.eval()
    with torch.no_grad():
      f_raw = self.encoder(x)
      f_before_bn = self.bottleneck(f_raw)
      f_after_bn = self.bnneck(f_before_bn)
    if was_training:
      self.train()
    return f_before_bn, f_after_bn

  def shared_step(self, batch, mode="train"):
    (x_i, x_j), labels = batch

    # 兩個視圖疊進同一個 batch，一起 forward
    x_cat = torch.cat([x_i, x_j], dim=0)
    cat_labels = torch.cat([labels, labels], dim=0)

    f_before_bn, f_after_bn, logits = self(x_cat)

    # 計算 SupCon Loss (約束特徵距離，使用 BN 前的特徵)
    loss_supcon = self.criterion_supcon(f_before_bn, cat_labels)

    # 計算 ID Loss (建立類別邊界，使用 BN 後過 Classifier 的 Logits)
    loss_id = self.criterion_id(logits, cat_labels)

    # 聯合 SupCon Loss 與 ID Loss (權重常為 1:1)
    loss = loss_supcon + loss_id

    self.log(f"{mode}_loss_supcon", loss_supcon, on_step=False, on_epoch=True, prog_bar=True)
    self.log(f"{mode}_loss_id", loss_id, on_step=False, on_epoch=True, prog_bar=True)
    self.log(f"{mode}_loss_total", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def training_step(self, batch, batch_idx):
    return self.shared_step(batch, mode="train")
        
  def on_validation_epoch_start(self):
    self._val_embeds_before, self._val_embeds_after, self._val_labels = [], [], []

  def validation_step(self, batch, batch_idx):
    images, labels = batch
    emb_before, emb_after = self.extract_feature(images)
    self._val_embeds_before.append(emb_before.float().cpu())
    self._val_embeds_after.append(emb_after.float().cpu())
    self._val_labels.append(labels.cpu())
    
  def _compute_rank1_map(self, query_emb, query_lbl, gallery_emb, gallery_lbl):
    q = F.normalize(query_emb, dim=1)
    g = F.normalize(gallery_emb, dim=1)
    sim = q @ g.T                       # [num_query, num_gallery]

    # ---- rank-1 ----
    top1_idx = sim.argmax(dim=1)
    top1_pred = gallery_lbl[top1_idx]
    rank1_acc = (top1_pred == query_lbl).float().mean().item()

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
    coverage = len(APs) / sim.size(0)
    return rank1_acc, mAP, coverage
    
  def on_validation_epoch_end(self):
    query_emb_before = torch.cat(self._val_embeds_before, dim=0)
    query_emb_after = torch.cat(self._val_embeds_after, dim=0)
    query_lbl = torch.cat(self._val_labels, dim=0)

    gallery_embeds_before, gallery_embeds_after, gallery_labels = [], [], []
    with torch.no_grad():
      for imgs, lbls in self.gallery_loader:
        imgs = imgs.to(self.device)
        emb_before, emb_after = self.extract_feature(imgs)
        gallery_embeds_before.append(emb_before.float().cpu())
        gallery_embeds_after.append(emb_after.float().cpu())
        gallery_labels.append(lbls)
    gallery_emb_before = torch.cat(gallery_embeds_before, dim=0)
    gallery_emb_after = torch.cat(gallery_embeds_after, dim=0)
    gallery_lbl = torch.cat(gallery_labels, dim=0)

    rank1_before, mAP_before, coverage_before = self._compute_rank1_map(
        query_emb_before, query_lbl, gallery_emb_before, gallery_lbl
    )
    rank1_after, mAP_after, coverage_after = self._compute_rank1_map(
        query_emb_after, query_lbl, gallery_emb_after, gallery_lbl
    )

    self.log("val_rank1_before_bn", rank1_before, prog_bar=True)
    self.log("val_mAP_before_bn", mAP_before, prog_bar=True)
    self.log("val_rank1_after_bn", rank1_after, prog_bar=False)
    self.log("val_mAP_after_bn", mAP_after, prog_bar=True)
    self.log("val_mAP_coverage", coverage_before, prog_bar=False) # coverage_before 跟 coverage_after 理論上會一樣，擇一紀錄即可

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
      optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=max(self.trainer.max_epochs - warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[warmup_epochs]
    )
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

# 獲取訓練集的狗狗總類別數，用於 ID Loss
num_classes = len(dataset.classes) 

for k, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
  train_dataset = TransformSubset(dataset, train_idx, DualViewTransform(train_transform))
  valid_dataset = TransformSubset(dataset, val_idx, valid_transform)          # 單視圖，作為用戶上傳的圖片
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
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, drop_last=False)
  gallery_loader = DataLoader(gallery_dataset, batch_size=256, num_workers=8, drop_last=False)

# ==========================================
# 5. 開始訓練
# ==========================================
  print("===============Initialize Model===============")
  
  model = DualLossModel(
    num_classes=num_classes,
    feat_dim=output_dim
  )
  model.gallery_loader = gallery_loader
  
  # 將 valid 結果最好的那組參數保留下來，分別存下 BN 前的和 BN 後的特徵向量，方便比較
  checkpoint_cb_before = ModelCheckpoint(
    monitor="val_mAP_before_bn", mode="max", save_top_k=1,
    dirpath=f"checkpoints/fold{k}",
    filename="best-before-{epoch}-{val_mAP_before_bn:.4f}"
  )
  checkpoint_cb_after = ModelCheckpoint(
    monitor="val_mAP_after_bn", mode="max", save_top_k=1,
    dirpath=f"checkpoints/fold{k}",
    filename="best-after-{epoch}-{val_mAP_after_bn:.4f}"
  )
  
  logger = TensorBoardLogger(save_dir=f"lightning_logs/DualLoss_{dataset_path.split('/')[0]}/", name=f"fold{k}")
  
  trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    precision=16,
    max_epochs=max_epochs,
    log_every_n_steps=10,
    callbacks=[checkpoint_cb_before, checkpoint_cb_after],
    enable_progress_bar=False,
    logger=logger
  )
  
  print("===============Current Parameter===============")
  print(f"batch_size = {batch_size}")
  print(f"max_epochs = {max_epochs}")
  print(f"output_dim = {output_dim}")
  print(f"part = {part}")
  
  trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
  print(f"Best model (before BN) saved in: {checkpoint_cb_before.best_model_path}")
  print(f"Best model (after BN)  saved in: {checkpoint_cb_after.best_model_path}")