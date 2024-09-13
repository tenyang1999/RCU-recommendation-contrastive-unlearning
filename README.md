# RCU：Recommendation Contrastive Unlearning

#### Authors: 
- [Tzu-Hsuan Yang]() (tzuhsuan1010@gmail.com)

## Overview 
This repository contains the code to preprocess datasets
<p align="center">
    <img src="overall_framework.png" width="1000" align="center">
</p>

## Datasets
Please run the following command to do train-test split. 
preprae_dataset.py for MovieLen100K, Douban, MovieLens1M. 
preprae_dataset_deg.py for AmazonBook. 
```
preprae_dataset.py
preprae_dataset_deg.py
```
## Experimental setups-

```python
pip install numpy 
pip install pandas
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## 使用方法
- ADEdataset.py為ADE的dataset讀取型態
- VOC_detect.ipynb為利用VOC進行Object Detection
- ADE_seg.ipynb為利用ADE進行Semantic Segmentation
- dual_task.ipynb為將兩者進行整合同時訓練
- utils為放置訓練以及預測會用到的工具們
