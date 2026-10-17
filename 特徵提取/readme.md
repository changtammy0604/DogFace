# **特徵提取**

這是第一版 SimCLR + Sibetan 資料集的說明，其他版本請在這裡跳轉：
### **[第二版 - SupCon + YT-BB-Dog](https://github.com/changtammy0604/DogFace/tree/SupCon%2BYT-BB-Dog-%E7%AC%AC%E4%BA%8C%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**
### **[第三版 - SupCon + dogFaceID](https://github.com/changtammy0604/DogFace/tree/SupCon%2BdogFaceID-%E7%AC%AC%E4%B8%89%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**
### **[第四版 - DualLoss + dogFaceID](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**

## **資料集**
**[Sibetan 資料集：](https://www.lirmm.fr/YT-BB-Dog_Sibetan/)**　

由印尼的研究團隊蒐集的資料集，涵蓋 59 隻個體與 1,755 張照片，解析度 720x480。

![image](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202026-09-23%20100145.png)

圖一、資料集處理示意圖，經過裁切後的身體與臉部分別送入不同的模型進行特徵提取。

## **模型架構**
SimCLR (Simple Framework for Contrastive Learning of VisualRepresentations) 是由 Google Brain 團隊於 2020 年提出的一種自監督學習 (Self-Supervised Learning) 框架。它在不需要任何人工標籤的情況下，透過對比學習 (Contrastive Learning) 讓模型學會抽取圖片特徵，並在ImageNet 的線性評估上達到了媲美監督學習的效果。

SimCLR 架構主要能被拆成四部分：

(1) 隨機資料增強：
對輸入的任意一張圖片隨機應用兩種不同的資料增強組合，生成兩張相互關聯的圖像，這組影像在論文中被稱作正樣本對 (PositivePair) ，兩張圖片分別用 xi 與 xj 表示。論文中指出，其中兩種增強方式的組合對最終模型的效果有顯著提升，其一是隨機裁剪及調整大小，可使模型理解整體與局部的關係；其二是隨機色彩抖動 (Random Color Jittering) ，避免模型只學到圖像中的顏色分布，而學不到更深層的語義結構。

(2) 基礎編碼器：
將經過資料增強後的圖片 𝑥̃𝑖 和 𝑥̃𝑗 透過編碼器 𝑓(⋅) 提取出特徵向量 ℎ𝑖，論文中使用的骨幹網絡 (Backbone) 為 ResNet-50：

$$ℎ_𝑖 = 𝑓(𝑥̃_𝑖) = 𝑅𝑒𝑠𝑁𝑒𝑡(𝑥̃_𝑖)$$

(3) 投影頭：
骨幹網絡所產生的結果為一組特徵向量，以往的模型通常直接用這組特徵向量計算損失 (loss) ，但 SimCLR 在骨幹網絡之後接上被稱為投影頭 (Projection Head) 𝑔(⋅) 的小型非線性神經網絡，在論文中是由兩層多層感知機器配上 ReLU 激活函數組成，用於將特徵向量映射到另一個低維空間 𝑧𝑖 中計算損失：

$$ 𝑧_𝑖 = 𝑔(ℎ_𝑖) = 𝑊^{(2)}𝜎(𝑊^{(1)}ℎ_𝑖)$$

其中 σ 為非線性激活函數 ReLU。而論文中以實驗證實，在訓練完成後丟棄投影頭、僅使用編碼器所產生的向量 ℎ 進行下游任務，如：分類、物件偵測 ，會比使用投影頭產生的 z 效果更好。文中指出是由於 𝑧 為了最小化損失，遺失了如顏色、形狀等有用資訊，反而 ℎ 保留了更豐富的語義。

(4) 損失函數：
架構中使用 NT-Xent (Normalized Temperature-scaled CrossEntropy) 損失函數計算損失，與同批次的樣本比較，縮短正樣本間的距離，並拉遠負樣本的距離。假設該批次有 N 張圖片，經過增強後產生 2N 個樣本。對於某正樣本對 (𝑖,𝑗)，其損失函數定義為：

$$ℒ_{𝑖,𝑗} = − log \frac{exp(sim(𝐳_𝑖, 𝐳_𝑗)/𝜏))}{\sum_{𝑘=1}^{2𝑁}𝟙[𝑘≠𝑖] exp(sim(𝐳_𝑖, 𝐳_𝑘)/𝜏))}$$

𝑠𝑖𝑚(𝑧_𝑖, 𝑧_𝑗) 表示兩個向量的餘弦相似度 (Cosine Similarity) ；
𝜏 作為溫度參數，用於調整模型對困難負樣本的處罰力度；
分母表示該樣本與批次中其餘 2N-1 個樣本的相似度，這些樣本被視作負樣本。
NT-Xent 的分母使其能依靠負樣本間互相的斥力以避免維度坍塌，即不論輸入圖像為何，模型都只會輸出相同的向量，因為當向量相同時，正樣本之間的距離為 0，滿足分子：
$$𝑒𝑥𝑝(𝑠𝑖𝑚(𝑧_𝑖, 𝑧_𝑗)/𝜏)$$
而分母部分則阻止模型透過這種方式繞過損失。也因此，SimCLR 依賴於批次大小所帶來的負樣本數，負樣本越多，越能約束模型將不同圖片的特徵向量分布到整個特徵空間中，訓練效果也就更好。

然而，NT-Xent 只將單一張照片所產生的兩個樣本視為要拉近的對象，並不會將單一個體的全部照片都視為同一類，因此當個體有超過一張照片時，就會導致同一個個體之間被拉遠，這並不符合我們的預期。

![image](https://github.com/changtammy0604/DogFace/blob/SimCLR%2BSibetan-%E7%AC%AC%E4%B8%80%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/images.png)

圖二、圖片來自 Simple Framework for Contrastive Learning of Visual Representations，對一張照片隨機取兩塊區域，透過 NT-Xent Loss 將他們拉近
