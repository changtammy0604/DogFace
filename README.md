# **結合深度學習與物件偵測之走失犬隻身分辨識與智慧協尋系統**

## **摘要**
　　隨著寵物飼養風氣盛行，流浪動物與寵物走失問題日益嚴重。然現行晶片掃描受限於接觸風險，社群協尋則多仰賴社群網路的人力轉發與肉眼比對，效率低落且易因主觀判斷產生誤差。本研究旨在開發一套走失犬隻身分辨識系統，利用電腦視覺與深度學習技術，解決跨鏡頭、不同角度下的犬隻特徵比對問題。

　　本研究將分為三個階段進行：首先，利用 YOLO 物件偵測模型，針對複雜背景下的犬隻進行定位與影像裁切；其次，導入 ResNet-50 作為骨幹網路，結合孿生神經網路 (Siamese Network) 與三元組損失函數 (Triplet Loss) 架構，進行細粒度 (Fine-graind) 之特徵提取。最後，建置向量檢索資料庫，實現使用者上傳圖片即可快速計算相似度並回傳比對結果的應用服務。本研究最終針對多角度輸入影像實現高準確度之檢索，並輸出相似度最高的犬隻臉部與身體前十名候選結果供人工確認。
  
-------------------------------------------------------
## **系統架構**
　　參考自 Deep Learning Pet Identification Using Face and Body (MDPI2023)，輸入犬隻圖片先經過裁剪，裁出個體與臉部區域並校正臉部角度，再進入特徵提取器，分別取得臉部與全身的特徵向量，找出資料庫內最相像的臉部與全身各十名。
![image](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/version3.jpg)

圖 1：匹配系統之架構圖

### **[一、偵測與裁切](https://github.com/changtammy0604/DogFace/tree/main/%E5%81%B5%E6%B8%AC%E8%88%87%E8%A3%81%E5%88%87)**

### **[二、特徵提取](https://github.com/changtammy0604/DogFace/tree/main/%E7%89%B9%E5%BE%B5%E6%8F%90%E5%8F%96)**

### **[三、身分識別與向量檢索](https://github.com/changtammy0604/DogFace/tree/DualLoss%2BFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/%E8%BA%AB%E5%88%86%E8%AD%98%E5%88%A5%E8%88%87%E5%90%91%E9%87%8F%E6%AA%A2%E7%B4%A2)**
### **[四、網站呈現](https://github.com/changtammy0604/DogFace/blob/DualLoss%2BFaceID-%E7%AC%AC%E5%9B%9B%E7%89%88/version3.jpg)**
-------------------------------------------------------
