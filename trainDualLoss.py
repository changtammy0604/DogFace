#!/usr/bin/env python
# coding: utf-8

# ==========================================================
# 用於訓練"SupCon+CrossEntropy"
# ==========================================================

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler

# ==========================================
# 1. 參數設定
# ==========================================
batch_size = 128
max_epochs = 200
output_dim = 2048
input_height = 224
part = "face"      # 模型是用於處理body/face
warmup_epochs = 10 # 用於 warmup, 由一開始小學習率逐漸長成設定的學習率，避免震盪

# NVIDIA GeForce RTX 3090 可以犧牲一點精度來換效能
torch.set_float32_matmul_precision('high')

# 訓練集&驗證集路徑
train_path = f"dataset/dogfaceid_{part}/train/"
valid_path = f"dataset/dogfaceid_{part}/valid/"

# ==========================================
# 2. 定義適合「個體辨識」的雙視圖數據增強
# ==========================================
class DualViewTransform:
    """將一張圖片轉換為兩個隨機增強版本 (適用於 Contrastive Learning)"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# 訓練期增強：保留裁切與翻轉，但降低顏色抖動，保留狗狗毛色！
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
    root=train_path,
    transform=DualViewTransform(train_base_transform) # 回傳 ((img_i, img_j), label)
)

valid_dataset = ImageFolder(
    root=valid_path,
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
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

# ==========================================
# 4. 定義 SupCon Lightning 模型
# ==========================================
class SupConReIDModel(pl.LightningModule):
    def __init__(self, num_classes, backbone='resnet50', feat_dim=2048, temperature=0.1):
        super().__init__()
        self.save_hyperparameters()
 
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

    def shared_step(self, batch, mode="train"):
        (x_i, x_j), labels = batch
 
        # 兩個視圖疊進同一個 batch，一起 forward
        x_cat = torch.cat([x_i, x_j], dim=0)
        cat_labels = torch.cat([labels, labels], dim=0)
 
        f_before_bn, f_after_bn, logits = self(x_cat)
 
        # 計算 SupCon Loss (約束特徵距離，使用 BN 前的特徵)
        loss_supcon = self.criterion_supcon(f_before_bn, cat_labels)
 
        # 計算 ID Loss (建立全域類別邊界，使用 BN 後過 Classifier 的 Logits)
        loss_id = self.criterion_id(logits, cat_labels)
 
        # 聯合 SupCon Loss 與 ID Loss (權重常為 1:1)
        loss = loss_supcon + loss_id

        self.log(f"{mode}_loss_supcon", loss_supcon, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_loss_id", loss_id, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_loss_total", loss, on_step=True, on_epoch=True, prog_bar=True)
 
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, mode="train")
        
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, mode="val")

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
        
        
    def extract_feature(self, x):
        """做檢索比對時用這個：只回傳 BN 後的特徵，並強制 eval 模式。"""
        was_training = self.training
        self.eval()
        f_raw = self.encoder(x)
        f_before_bn = self.bottleneck(f_raw)
        f_after_bn = self.bnneck(f_before_bn)
        if was_training:
            self.train()
        return f_after_bn

# ==========================================
# 5. 開始訓練
# ==========================================
print("=============== 初始化模型 ===============")
# 動態獲取訓練集的狗狗總數，用於ID Loss
num_classes = len(train_dataset.classes)

model = SupConReIDModel(
    num_classes=num_classes,
    feat_dim=output_dim
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    precision="16",
    max_epochs=max_epochs,
    log_every_n_steps=10
)

print("=============== 目前參數 ===============")
print(f"batch_size = {batch_size}")
print(f"max_epochs = {max_epochs}")
print(f"output_dim = {output_dim}")
print(f"part = {part}")

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)