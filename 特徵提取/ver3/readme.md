# **特徵提取**

這是第三版 DualLoss + dogFaceID 資料集的說明，其他版本請在這裡跳轉：
### **[第一版 - SimCLR + Sibetan](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/ver1)**
### **[第二版 - SupCon + YT-BB-Dog](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/ver2)**
### **[第四版 - DualLoss + dogFaceID](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/ver4)**

## **資料集**
**[dogfaceid 資料集：](https://drive.google.com/file/d/1UAnR1zRmme_WTK1TecGdtYaV5zmB-jKp/view?usp=sharing)**　

專為本專案蒐集的犬隻資料集，以台灣常見犬種為主。包含 dogfaceid_face 及 dogfaceid_body 兩部分，分別為擷取出的臉部及身體。兩部分裡面又分為 train_valid_all 以及 test，分別用於訓練與驗證，以及測試。
資料集收錄 106 隻個體，770 張全身照片，682 張臉部照片。(因為有些照片中的臉部被遮擋，無法擷取，所以臉部照片數量會比較少)

![image](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202026-09-23%20100145.png)

圖一、資料集處理示意圖，經過裁切後的身體與臉部分別送入不同的模型進行特徵提取。

## **實作評估方式**

在驗證階段，採用以下方式評估匹配效果：

(一) 採用 5 倍交叉驗證 (K-Fold Cross-Validation)

在資料量較小的情況下，神經網路極易產生過度擬合，且單一次的資料集切割可能因偶然性導致效能評估失真。 因此，為確保模型架構演進的客觀性，本專題在所有架構版本的訓練與評估中，均導入了五倍交叉驗證。透過將資料集均分為五個互不重疊的子集，進行五回合的輪替訓練與驗證，以驗證系統的泛化能力，確保每一次架構更改都具備統計意義。

(二) 採用平均精度均值 (Mean Average Precision, mAP) 進行驗證

在驗證階段將整個訓練集構建為檢索庫 (Gallery)。系統會將驗證集中的每張查詢影像提取特徵後，與檢索庫中的所有影像進行相似度比對與名次排序，計算本輪的平均精度均值。

## **模型架構**
基於上一版，加入 5 倍交叉驗證與計算平均精度均值的步驟，也將資料集改為我們自製的 dogfaceid。



