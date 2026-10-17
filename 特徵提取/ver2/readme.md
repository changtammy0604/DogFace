# **特徵提取**

這是第二版 SupCon + YT-BB-Dog 資料集的說明，其他版本請在這裡跳轉：
### **[第一版 - SimCLR + Sibetan](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/ver1)**
### **[第三版 - SupCon + dogFaceID](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/ver3)**
### **[第四版 - DualLoss + dogFaceID](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/ver4)**

## **資料集**
**[YT-BB-Dog 資料集：](https://www.lirmm.fr/YT-BB-Dog_Sibetan/)**　

圖片來源自 Youtube 上的影片截圖，涵蓋 2,723 隻狗與 27,036 張照片，但時常出現鏡頭或個體晃動造成的模糊。

![image](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BdogFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202026-09-23%20100145.png)

圖一、資料集處理示意圖，經過裁切後的身體與臉部分別送入不同的模型進行特徵提取。

## **實作評估方式**

只記錄驗證損失，但我們很快發現這不能很好的呈現最終的匹配效果。

## **模型架構**

為解決避免互相推斥同類樣本，本研究將架構更改為監督式對比學習。SupCon 引入了標籤資訊，在計算損失時，不再只依賴單張圖片的擴增視角，而是將批次所有標籤相同的照片都視為正樣本互相拉近，不同標籤的照片則全數推遠，使 SimCLR 不再將同隻狗的照片推開，而是為每個犬隻個體建立高密度的叢集 (Cluster)，更適合重識別系統。
架構上沿用投影層、隨機資料增強、ResNet-50 骨幹，將損失函數由 NT-Xent 改為 SupCon。

### **損失函數**
SupCon 與 NT-Xent 原理類似，皆是將相同類別拉近、不同類別拉遠，不同的是 SupCon 會看類別標籤，因此只須將個體的照片放入同一類別，就能避免錯誤的將相同個體拉遠，公式如下：

$$L_{out}^{sup} = \sum_{𝑖∈𝐼}\frac{−1}{|𝑃(𝑖)|}\sum_{𝑝∈𝑃(𝑖)}log \frac{exp(𝑠𝑖𝑚(𝑧_𝑖, 𝑧_𝑝)/𝜏)}{\sum_{𝑎∈𝐴(𝑖)}exp(𝑠𝑖𝑚(𝑧_𝑖, 𝑧_𝑎)/𝜏)}$$

其中，P(i) 為批次中與樣本 i 具備相同標籤的樣本集合。
若輸入完全隨機的資料，批次內可能沒有同一隻狗的照片，導致公式中的 |P(i)| 為零，因此本研究透過 pytorch_metric_learning.samplers 導入 MPerClassSampler 並設定 m = 4，強制保證每個批次內只要抽中某隻狗，就必定會包含 4 張該犬隻的照片，確保了正樣本的穩定供給；此外，透過torchvision.transforms.ColorJitter 調整，本研究降低了顏色破壞，避免毛色與斑紋等犬隻特徵被過度擾動。

