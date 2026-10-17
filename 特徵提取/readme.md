# **特徵提取**

這是第一版 SimCLR + Sibetan 資料集的說明，其他版本請在這裡跳轉：
### **[第二版 - SupCon + YT-BB-Dog](https://github.com/changtammy0604/DogFace/tree/SupCon%2BYT-BB-Dog-%E7%AC%AC%E4%BA%8C%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**
### **[第三版 - SupCon + dogFaceID](https://github.com/changtammy0604/DogFace/tree/SupCon%2BdogFaceID-%E7%AC%AC%E4%B8%89%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**
### **[第四版 - DualLoss + dogFaceID](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**

## **資料集**
**[dogfaceid 資料集：](https://drive.google.com/file/d/1UAnR1zRmme_WTK1TecGdtYaV5zmB-jKp/view?usp=sharing)**　

專為本專案蒐集的犬隻資料集，以台灣常見犬種為主。包含 dogfaceid_face 及 dogfaceid_body 兩部分，分別為擷取出的臉部及身體。兩部分裡面又分為 train_valid_all 以及 test，分別用於訓練與驗證，以及測試。
資料集收錄 106 隻個體，770 張全身照片，682 張臉部照片。(因為有些照片中的臉部被遮擋，無法擷取，所以臉部照片數量會比較少)

![image](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202026-09-23%20100145.png)

圖一、資料集處理示意圖，經過裁切後的身體與臉部分別送入不同的模型進行特徵提取。

## **模型架構**
結合 SimCLR 和 SupCon 上學到的對比學習，與 Visible Thermal Person Re-Identification via Dual-Constrained Top-Ranking、Person Re-Identification Method Based on Dual Descriptor Feature Enhancement 及 Person Re-Identification Based on Joint Loss and Multiple Attention Mechanism 等多篇與行人重識別相關文獻所使用的雙重損失函數，並參考 Bag of Tricks and A Strong Baseline for Deep Person Re-identification 文中提出的，在三元組損失函數與身分損失函數之間加入批次正規化，以避免兩個損失函數的衝突。

![image](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/train_%E6%9E%B6%E6%A7%8B.jpg)

圖二、使用雙重損失函數後的架構。參考 Bag of Tricks and A Strong Baseline for Deep Person Re-identification 文中的主張，在 SupCon Loss 與 Cross-Entropy Loss 之間加入批次正規化。由於損失函數已由 SimCLR 所使用的 NT-Xent loss 改為 SupCon Loss，因此除了圖片本身產生的 $$x_i$$ 和 $$x_j$$ 會互相吸引，與同個類別產生之 $$x_i$$ 和 $$x_j$$ 皆相吸，並與其他類別相斥。

本研究還實作了線性預熱 ，讓學習率在前 10 個 Epoch 緩步上升，隨後才進入餘弦退火 (Cosine Annealing) 衰減，確保預訓練權重不被瞬間的巨大誤差梯度破壞。

### **雙重損失函數訓練**

雙重損失函數訓練有以下幾種優勢：
1. 提升模型泛化能力
2. 穩定訓練過程，避免梯度爆炸、消失

本次使用的是三元組損失函數搭配身分損失函數的方式，三元組損失函數負責將相同個體聚合，不同個體拉遠；身分損失函數在不同類別間劃定邊界，但不保證資料緊湊度。三元組損失函數沿用 SupCon Loss，身分損失函數則使用 Crosss-entropy Loss。
但 SupCon Loss 與 Crosss-entropy Loss 要求的目標不同，直接訓練可能反而使參數難以收斂。因此 Bag of Tricks and A Strong Baseline for Deep Person Re-identification 提出應在兩個損失函數之間加入一層批次正規化層，將特徵空間分開，使損失函數能各自調整特徵空間的而不受彼此干擾。

### **實作評估方式**

在驗證階段，採用以下方式評估匹配效果：

(一) 採用 5 倍交叉驗證 (K-Fold Cross-Validation)

在資料量較小的情況下，神經網路極易產生過度擬合，且單一次的資料集切割可能因偶然性導致效能評估失真。 因此，為確保模型架構演進的客觀性，本專題在所有架構版本的訓練與評估中，均導入了五倍交叉驗證。透過將資料集均分為五個互不重疊的子集，進行五回合的輪替訓練與驗證，以驗證系統的泛化能力，確保每一次架構更改都具備統計意義。

(二) 採用平均精度均值 (Mean Average Precision, mAP) 進行驗證

在驗證階段將整個訓練集構建為檢索庫 (Gallery)。系統會將驗證集中的每張查詢影像提取特徵後，與檢索庫中的所有影像進行相似度比對與名次排序，計算本輪的平均精度均值。

### **[訓練好的參數檔](https://drive.google.com/file/d/1Jte_1NpHfV5YykD8BdFO7ny6Kb2Q7Qtx/view?usp=sharing)**
