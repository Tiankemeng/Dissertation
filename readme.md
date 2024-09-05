## 基于Bert在问答数据集上的元学习算法探究

### 数据集
本项目使用个人自建数据(data/ipqa)和wikiQA(data/wikiQA)数据集两个数据对元学习算法进行评测。

### 模型
本项目使用Bert预训练模型(pretrained_models/bert-base-uncased)作为基础模型进行算法探究实验。

### 算法
本项目评测的元学习算法包括maml，原型网络(pn)以及reptile算法。算法代码在engine_train.py文件中。

### 评测指标
本项目使用MAP和MRR两个常用评价指标，计算代码在metrics.py文件中。

### 代码运行
代码运行命令在run.sh中，运行单行命令则为跑对应数据集的对应算法，直接运行run.sh可以的到三个算法在2个数据集的结果，并画出对比图。结果保存在results中，对比图在results/lr_1e-5_ep50/summary文件夹中。

### 代码运行命令
`bash run.sh`