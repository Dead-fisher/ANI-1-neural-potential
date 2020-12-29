# ANI-1-neural-potential
ANI-1 neural potential for energy predicting

#### 12.30
实现了使用ani1_gdb10_ts.h5数据集小规模训练，结果储存在logs中。
TODO
优化损失函数
修改batchsize的提供方式，提高batchsize到1000左右
确定训练集和测试集，在训练集上预测试
在Linux系统预训练


#### 12.29
封装了部分函数
修改了数据类型为float

```
train_ene = torch.tensor(energies[:upper_bound], dtype=torch.float)
test_ene = torch.tensor(energies[upper_bound:], dtype=torch.float)
```

#### 12.28
网络基本框架
TODO

#### 12.27
实现dataloader
