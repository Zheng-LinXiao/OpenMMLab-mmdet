# OpenMMLab-mmdet-basic
## 使用环境
win10 + cuda11.1 + cudnn8.0 +torch1.8.0

## 数据准备
下载数据集并按如下方式组织数据集：
```shell
mmdetection
├── data
│   ├── balloon
│   │   ├── anna
│   │   │    ├──train.json
│   │   │    └──val.json
│   │   ├── train
└   └   └── val
```
# 训练
训练脚本
```bash
python tools/train.py mask_rcnn_r50_rpn_1x.py --work-dir work/mask_rcnn_r50_fpn_1x
```
# 结果
| bbox AP | mask AP |
| :----:  | :-----: |
|  76.73  |  79.29  |
