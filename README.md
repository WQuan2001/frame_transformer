# frame_transformer

![The network architecture of Pyraformer.](https://github.com/Dmovic/frame_transformer/blob/main/img/Figure_1.png)

<center><b>Figure 1.</b> Pyraformer的网络结构</center>

- 输入：原数据、协变量、位置编码
- CSCM：构建多分辨率 C 叉树（每个叉表示一个分辨率尺度）
- PAM：利用注意力机制时序建模
- 根据任务不同，设置不同的Prediction Strategy

## attention

![The Pyramidal Attention Mechanism.](https://github.com/Dmovic/frame_transformer/blob/main/img/Figure_2.png)

如图所示，利用金字塔图以多分辨率方式描述观测时间序列的时间依赖关系。我以将金字塔图分解为两个部分：尺度间连接和尺度内连接。尺度间连接形成一棵 Cary 树，其中每个父节点有 C 个子节点。如果我们将金字塔图的最细尺度与原始时间序列的小时观测值联系起来，那么较粗尺度的节点就可以被视为时间序列的日、周甚至月特征。因此，金字塔图提供了原始时间序列的多分辨率表示。

此外，只需通过尺度内连接将相邻节点连接起来，就能更容易地捕捉到较粗尺度中的长距离依赖关系（如月依赖关系）。换句话说，较粗尺度有助于以图形化的方式描述长程相关性，其简洁性远远超过单一的最细尺度模型。

## introduce

现有的方法不能同时满足高准确率和低时间复杂度：

- 基于CNN和RNN的方法：时间复杂度低，准确率低
- 基于transformer的方法：准确率高，时间复杂度高

构建C叉树，利用时间序列的多分辨率表示，捕捉各种时间依赖性。提出注意力金字塔模块（PAM）。

- 提出inter-scale tree structure概念：对不同分辨率的尺度进行序列建模
- 提出intra-scale neighboring connections概念：在不同时间范围内对时间依赖性建模

## dataset

Wind：该数据集包含1986年至2015年间28个国家每小时的能源潜力估计值，以发电厂最大产量的百分比表示。与剩余的数据集相比，它更稀疏，并且周期性地呈现出大量的零。由于这个数据集的规模很大，训练集和测试集之间的比率大约为32:1。

App Flow：该数据集由Ant Group收集。它包含部署在16个逻辑数据中心的128个系统的每小时最大流量，总共产生1083个不同的时间序列。每个系列的长度超过4个月。每个时间序列分为两个部分，分别进行训练和测试，比例为32:1。

Electricity（Yu等人，2016）：该数据集包含370个用户每15分钟记录的耗电量时间序列。根据DeepAR（Salinas等人，2020），我们每4个记录汇总一次，以获得每小时的观测结果。该数据集用于单步和长期预测。我们使用2011-01-01至2014-09-01的数据进行单步预测，并使用2011-04-01至2014-04-01的数据进行长期预测。

ETT（Zhou等人，2021）：该数据集包含从2个站点收集的2台变压器的2年数据，包括油温和6个电力负荷特征。提供每小时（即ETTh1）和每15分钟（即ETTm1）的观测。该数据集通常用于长期预测的模型评估。在这里，我们跟踪了Informer（Zhou等人，2021），并将数据分为12个月和4个月，分别用于训练和测试。


### 电力变压器数据集 (ETDataset)

提供了涉及电力变压器的多个数据集，用于支撑”长时间序列”相关的研究。所有的数据都经过了预处理，并且以`.csv`的格式存储。这些数据的时间跨度为2016年7月到2018年7月。

*数据列表*

- [x] **ETT-small**：含有2个电力变压器（来自2个站点）的数据，包括负载、油温。
- [ ] **ETT-large**：含有39个电力变压器（来自39个站点）的数据，包括负载、油温。
- [ ] **ETT-full**：含有69个电力变压器（来自39个站点）的数据，包括负载、油温、位置、气候、需求。


### 为什么引入 *油温数据* 到该数据集中？

电力分配问是电网根据顺序变化的需求管理电力分配到不同用户区域。但要预测特定用户区域的未来需求是困难的，因为它随工作日、假日、季节、天气、温度等的不同因素变化而变化。现有预测方法不能适用于长期真实世界数据的高精度长期预测，并且任何错误的预测都可能产生严重的后果。因此当前没有一种有效的方法来预测未来的用电量，管理人员就不得不根据经验值做出决策，而经验值的阈值通常远高于实际需求。保守的策略导致不必要的电力和设备折旧浪费。值得注意的是，变压器的油温可以有效反映电力变压器的工况。我们提出最有效的策略之一，是预测变压器的油温同时设法避免不必要的浪费。
为了解决这个问题，我们的团队与北京国网富达科技发展公司建立了一个平台并收集了2年的数据。我们用它来预测电力变压器的油温并研究电力变压器极限负载能力。

## ETT-small:

提供了两年的数据，每个数据点每分钟记录一次（用 *m* 标记），它们分别来自中国同一个省的两个不同地区，分别名为ETT-small-m1和ETT-small-m2。每个数据集包含2年 * 365天 * 24小时 * 4 = 70,080数据点。 此外，我们还提供一个小时级别粒度的数据集变体使用（用 *h* 标记），即ETT-small-h1和ETT-small-h2。 每个数据点均包含8维特征，包括数据点的记录日期、预测值“油温”以及6个不同类型的外部负载值。

具体来说，数据集中包含短周期模式，长周期模式，长期趋势和大量不规则模式。数据显示出了明显的季节趋势。为了更好地表示数据中长期和短期重复模式的存在，“油温”保持了一些短期的局部连续性，而其他的变量（各类负载）则显示出了短期的日模式（每24小时）和长期的周模式（每7天）。

数据集是使用`.csv`形式进行存储的，在图3中我们给出了一个数据的样例。其中第一行（8列）是数据头，包括了 "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL" 和 "OT"。

### Mindspore

使用鹏城实验室和华为联合开发的一款Mindspore生态适配工具——**MSadapter**。该工具能帮助用户高效使用昇腾算力，且在不改变**原有PyTorch用户**使用习惯的前提下，将代码快速迁移到Mindspore生态上。

MSAdapter的API完全参照PyTorch设计，用户仅需少量修改就能轻松地将PyTorch代码高效运行在昇腾上。目前MSAdapter已经适配**torch、torch.nn、torch.nn.function、torch.linalg**等800+接口；全面支持**torchvision**；并且在MSAdapterModelZoo中验证了70+主流PyTorch模型的迁移。

安装`msadapter`

```
pip install msadapter
```


安装`mindspore`

```
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0rc2/MindSpore/unified/x86_64/mindspore-2.3.0rc2-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```


使用Msadapter进行迁移

替换头文件部分导入内容

```python
import torch
import torch.optim as optim
from dataloader import *
```

为

```python
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.optim as optim
import mindspore as ms
from mindspore import Parameter，nn
from msadapter.pytorch.utils.data import *
```

替换神经网络优化器

```python
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr)
```

为

```python
ms_model_parameters = [x for x in model.parameters() if x.requires_grad]
optimizer = ms.nn.Adam(Parameters(ms_model_parameters), opt.lr)
```

替换评价指标

```python
criterion = torch.nn.MSELoss(reduction = 'None')
```

为

```python
criterion = nn.MSELoss(reduction = 'None')
```

### 实验

notice

1. 原文作者在Pytorch框架下利用4块GPU进行训练，本文仅在1块GPU上复现，需要手动修正
2. 原文作者在进行ETT1h数据集长时预测时，对神经网络层命名有误，需要重新命名确保神经元参数匹配

step

1. download dataset
2. data preprocess 
3. training 
4. load checkpoint 
5. evaluate

#### 环境

- Ubuntu OS
- Python 3.7
- CUDA 11.1
- TVM 0.8.0 (optional)

Dependencies can be installed by:

```
pip install -r requirements.txt
```

![preprocess data](https://github.com/Dmovic/frame_transformer/blob/main/img/preprocess%20data.png)

#### 数据准备

下载好数据集后按照如下目录结构处理。

```
${CODE_ROOT}
    ......
    |-- data
        |-- elect
            |-- test_data_elect.npy
            |-- train_data_elect.npy
            ......
        |-- flow
            ......
        |-- wind
            ......
        |-- ETT
            |-- ETTh1.csv
            |-- ETTh2.csv
            |-- ETTm1.csv
            |-- ETTm2.csv
        |-- LD2011_2014.txt
        |-- synthetic.npy
```

运行脚本构造数据。

```
python simulate_sin.py
```

#### 训练

long range forecasting

```
sh scripts/Pyraformer_LR_FC.sh
```

![training long predict](https://github.com/Dmovic/frame_transformer/blob/main/img/training%20long%20predict.png)

![long_range_main.py](https://github.com/Dmovic/frame_transformer/blob/main/img/256.png)

![[Elect]single_step_main.py -data_path -dataset elect -eval](https://github.com/Dmovic/frame_transformer/blob/main/img/eval.png)

single step forecasting

```
sh scripts/Pyraformer_SS.sh
```

![training short predict](https://github.com/Dmovic/frame_transformer/blob/main/img/training%20short%20predict.png)

![[flow]single_step_main.py -data_path -dataset flow -eval](https://github.com/Dmovic/frame_transformer/blob/main/img/flow.png)

![e256](https://github.com/Dmovic/frame_transformer/blob/main/img/e256.png)

![[wind]single_step_main.py -data_path -dataset wind -eval](https://github.com/Dmovic/frame_transformer/blob/main/img/wind.png)
