###  tmux

window 1: half mixup (原始0.5+shuffle后0.5)

window 2: 

### result

1. CAL baseline：

​	![CAL_base结果](D:\Workspace\Re-id\cloth-changing\feature mixup实验\CAL_base结果.png)

### 实验记录

不加入任何shuffle模块时：

​	CC：mAP=50.9%，top1=50.0%；SC：mAP=99.7%，top1=100% (25 epoch)

​	CC：mAP=52.6%, top1=49.8%；SC：mAP=99.8%, top1=100.0% (60epoch)

对于channel attention后相加的方法（SHUFFLE 1）

​	加入，不shuffle时：CC：mAP=53.4%, top1=52.8%；SC：mAP=99.8%，top1=100.0%

​	**训练shuffle，测试不shuffle：CC：mAP=49.3%，top1=46.6%；SC：mAP=99.3%，top1=99.7%**

​	训练、测试均shuffle：CC：mAP=48.8%，top1=46.2%；SC：mAP=99.3%，top1=99.9%

（决定采用训练时shuffle，测试时不shuffle的方法，效果最好，也比较符合数据增强的实现惯例）

先channel划分，再shuffle，最后拼接的方法（SHUFFLE2）

​	划分为两个channel=C/2---分别做channel attention---shuffle---concat：

​			不shuffle：CC：mAP=52.6%，top1=53.4%；SC：mAP=99.1%，top1=100%

​			shuffle：CC：mAP=52.4%，top1=49.1%；SC：mAP=99.9%，top1=100%

​	划分为两个channel=C/2---shuffle--concat：

​			不shuffle（等价于baseline）CC：mAP=52.6%, top1=49.8%；SC：mAP=99.8%, top1=100.0% (60epoch)

​			shuffle：CC：mAP=52.0%，top1=50.6%；SC：mAP=99.8%，top1=100% (60 epoch)

​	划分为两个channel=C/2---shuffle---分别做channel attention---concat：

​			不shuffle：CC：mAP=%，top1=%；SC：mAP=%，top1=%

​			shuffle：CC：mAP=53.9%，top1=52.7%；SC：mAP=99.8%，top1=100%（60 epoch）

​	**划分为两个channel=C/2---shuffle--concat---channel attention：**

​				不shuffle：CC：mAP=52.5%，top1=49.5%；SC：mAP=99.8%，top1=100% (60 epoch)

​				**shuffle：CC：mAP=55.3%，top1=53.3%；SC：mAP=99.9%，top1=100% (60 epoch)**





|                                     | 1    |      | 2    |      | 3    |      | 4    |      | 5    |      |
| ----------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|                                     | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  |
| baseline                            | 49.8 | 52.6 | 54.7 | 55.7 | 51.8 | 54.4 | 55.0 | 55.2 |      |      |
| shuffle                             | 50.6 | 52.0 | 50.8 | 52.6 | 54.0 | 54.8 | 48.6 | 50.9 |      |      |
| shuffle+保留原特征                  | 52.0 | 53.3 | 51.7 | 54.1 | 52.0 | 52.8 | 52.8 | 52.5 |      |      |
| shuffle+保留原特征+consistency loss | 50.7 | 52.8 | 52.3 | 53.0 | 51.5 | 52.9 | 52.0 | 52.9 |      |      |
| shuffle+channel att.                | 52.3 | 55.3 | 50.4 | 52.5 | 51.6 | 53.0 | 49.6 | 52.6 |      |      |
| channel att.+shuffle+保留原特征     | 50.8 | 51.7 | 50.9 | 51.6 | 50.8 | 52.1 | 50.2 | 50.6 |      |      |
| shuffle+保留+cloth classification   | 51.8 | 51.5 | 54.4 | 54.4 | 50.4 | 51.4 | 52.4 | 53.4 |      |      |

​	

shuffle+channel att.时，前1/2channel和后1/2channel的attention weight：

​	seed 1：0.5，0.5

​	seed 2：0.5，0.5

channel att.然后shuffle时，不加consistency loss，channel attention向量对应各通道的权重几乎全是0.5，感觉没有学到选择性。



- mixup layer加在不同层的效果对比：

|                | 1    |      | 2    |      | 3    |      | 4    |      | 5    |      |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|                | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  |
| baseline       | 49.8 | 52.6 | 54.7 | 55.7 | 51.8 | 54.4 | 55.0 | 55.2 |      |      |
| layer1-2       | 53.6 | 53.4 | 51.0 | 53.5 | 50.8 | 52.6 | 54.0 | 53.5 |      |      |
| layer2-3       |      |      |      |      |      |      |      |      |      |      |
| layer3-4       | 52.0 | 53.3 | 51.7 | 54.1 | 52.0 | 52.8 | 52.8 | 52.5 |      |      |
| layer4-clshead |      |      |      |      |      |      |      |      |      |      |

shuffle加在靠前的层时，shuffle前后样本的一致性损失较高。layer1-2之间，0.008左右；layer3-4之间，0.001左右。



- 分类器上加入衣物分类：
  1. 如果直接同时用feature做id分类和衣物分类，点数不涨。二者有可能形成对抗
  2. 如果直接用衣物id做分类：
  3. 将衣物id作为行人id的子标签，同一行人不同衣物之间的损失权重设的较小



- cloth mask辅助的patch shuffle方法：

  PRCC数据集：
  
  默认设置：上衣下衣都交换，在maxpool-layer1之间，patchsize=8

|                                       | 1    |      | 2    |      | 3    |      | 4    |      | 5    |      |
| ------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|                                       | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  |
| baseline                              | 50.8 | 51.6 | 53.1 | 54.5 | 55.6 | 54.1 | 53.0 | 54.2 |      |      |
| patch shuffle(only upperclothes)      | 56.1 | 57.8 | 58.0 | 59.9 | 58.2 | 59.5 | 55.9 | 57.3 |      |      |
| patch shuffle                         | 57.9 | 58.2 |      |      |      |      |      |      |      |      |
| patch shuffle(use old feat)           | 55.6 | 57.6 | 54.8 | 56.5 |      |      |      |      |      |      |
| baseline(original)                    | 49.8 | 52.6 | 54.7 | 55.7 | 51.8 | 54.4 | 55.0 | 55.2 |      |      |
| patch shuffle(original)               | 56.3 | 57.2 | 55.9 | 58.0 | 56.7 | 58.0 | 57.2 | 57.3 |      |      |
| patch shuffle(original)(use old feat) | 55.6 | 57.6 | 55.7 | 55.8 | 57.2 | 57.2 | 56.1 | 57.8 |      |      |
|                                       |      |      |      |      |      |      |      |      |      |      |
|                                       |      |      |      |      |      |      |      |      |      |      |

​		LTCC数据集：（每格内的三个数字，分别对应general/SC/CC setting）

​		默认设置：上衣和下衣都做交换，在maxpool-layer1之间，patchsize=8

|                          | 1              |                | 2              |                | 3              |                | 4              |                |
| ------------------------ | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
|                          | top1           | mAP            | top1           | mAP            | top1           | mAP            | top1           | mAP            |
| baseline                 | 72.4/77.7/37.5 | 36.7/58.8/16.5 | 74.8/79.9/38.0 | 37.6/58.7/17.3 | 72.2/78.4/37.0 | 37.2/58.4/16.9 | 71.4/77.5/37.0 | 37.7/58.8/17.2 |
| patch shuffle            | 72.6/77.0/39.0 | 36.2/54.0/18.2 | 70.8/78.2/37.2 | 36.7/56.3/17.8 | 70.0/76.3/38.8 | 36.8/56.0/17.8 | 71.4/76.3/39.0 | 37.4/58.0/17.9 |
| patch shuffle+保留原特征 | 71.8/78.2/39.3 | 36.8/56.5/17.7 | 72.0/77.0/39.0 | 36.9/56.9/17.4 | 73.2/78.9/40.3 | 37.2/57.6/17.7 | 72.4/78.9/38.3 | 37.8/58.5/18.6 |



- cross-attention实验

  默认设置：prcc dataloader不sort。d=1024，C=2048，ca里面加bn
  
  patchshuffle+ca代表同时利用原先的pooling后特征和cross-attention的特征计算loss

|                                                  | 1    |      | 2    |      | 3    |      | 4    |      | 5    |      |
| ------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|                                                  | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  | top1 | mAP  |
| patch shuffle (avgpool)(w/o ca)(use_old_feat)    | 52.9 | 53.8 |      |      |      |      |      |      |      |      |
| patch shuffle (avgpool)(w/o ca)                  | 58.3 | 56.4 |      |      |      |      |      |      |      |      |
| patch shuffle (avgpool  add ca)                  | 56.0 | 56.3 |      |      |      |      |      |      |      |      |
| patch shuffle + ca(id0.5)                        | 56.5 | 57.5 |      |      |      |      |      |      |      |      |
| patch shuffle + ca(id0.5)(use old feat)          | 56.0 | 54.2 |      |      |      |      |      |      |      |      |
| patch shuffle + ca(id0.5+cloth0.5)(use old feat) |      |      |      |      |      |      |      |      |      |      |





### 问题

- ~~channel切分感觉可能存在一个问题：模型会逐渐关注到前1/2的channel。因为consistency loss的权重设为0时，这个loss在训练过程中依然是在不断减小的。~~
- ~~mixup block放在最后一层是否合理？--可以试着往前放~~
- ~~怎样在最终任务上加loss，才能避免模型直接不学习shuffle那部分的特征了？~~
- prcc用shuffle前特征时，点数反而不如不用shuffle的。并且用shuffle前特征时，模型开始的epoch涨点很快，后面就不涨了。需要探究一下有没有可能是学习率的影响

### 可尝试的实验

- pooling方式：avg，max，maxavg（目前）
- shuffle前和shuffle后对应样本的一致性loss
- 背景mask掉，背景shuffle
- cross-attention模块的学习率较低时，似乎收敛更快，可以考虑cross-attention模块和backbone用不同的学习率