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

# 训练pth和结果视频
链接：https://pan.baidu.com/s/1hG_pcJlNWSADrZqtX8Bb_w?pwd=sfsx 
提取码：sfsx

# OpenMMLab-mmdet-advanced
## 使用环境
win10 + cuda11.1 + cudnn8.0 +torch1.8.0
这边注意！
使用的是mmyolo，并且numpy包需要<1.20.0。
因为在代码中使用到np.bool，而在numpy>=1.20.0中np.bool被删除，改成np.bool_
## 数据准备
下载数据集并按如下方式组织数据集：
```shell
mmdetection
├── data
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
└   └   └── VOCcode
```
# 训练
训练脚本
```bash
python tools/train.py voc.py --work-dir work/voc
```
# 结果
| mAP |
|:---:|
|0.887|

# 训练pth和结果视频
链接：https://pan.baidu.com/s/1q8ydadCbHT9zQtR7ATUjqg?pwd=i7yy 
提取码：i7yy
