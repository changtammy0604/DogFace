# **偵測與裁切**
　　實務場景中之政府或民間通報照片，僅提供未經處理之原始犬隻影像，缺乏可用於監督式學習 (Supervised Learning) 的幾何座標標註。為使YOLOv11-pose 模型具備精準偵測犬隻面部關鍵點之先驗能力，本研究導入了 Kaggle 平台上的公開視覺資料集 PetFaceDetection[7]作為模型微調(Fine-tuning) 的訓練基底。該訓練資料集專門針對寵物面部進行了標註，其標註維度契合本系統之需求，包含：

1. 面部邊界框 (Face Bounding Box)：定義犬隻面部之核心特徵區域，濾除不必要之背景雜訊。
2. 核心幾何錨點 (Facial Landmarks)：標註了雙眼 (左眼、右眼) 與鼻子之 2D 像素座標。
  
　　進行資料集整理過後，共有 12709 張標示完整之犬隻圖片，其中訓練集有9526 張，驗證集則有 3183 張。本研究利用此資料集對 YOLO11-pose 網路進行遷移學習 (Transfer Learning)，使模型學會跨犬種之頭骨結構與眼鼻相對位置。待模型訓練收斂並具備關鍵點擷取能力後，再將其應用至後續系統之犬隻圖片切割和特徵提取。

　　擷取出面部邊界框後，系統會透過核心幾何錨點計算出仿射矩陣，用於將臉部轉正以減少干擾。由於直接旋轉會導致圖片周邊出現黑框（如圖 2），因此旋轉後會做二次裁切（如圖 3），再以色彩填補的方式（如圖 4），將照片大小填補至設定的長寬。

![image](https://github.com/changtammy0604/DogFace/blob/main/version3.jpg)
圖 2：先以旋轉整張圖片，並以首次邊界框範圍進行延伸，取為正方形，有黑邊
![image](https://github.com/changtammy0604/DogFace/blob/main/version3.jpg)
圖 3：先旋轉整張圖片並圖片延伸補黑邊，再次偵測邊界框並切割，產生正方形成果
![image](https://github.com/changtammy0604/DogFace/blob/main/version3.jpg)
圖 4：先旋轉整張圖片，二次偵測邊界框後切割，犬隻臉部有嚴重拉伸
