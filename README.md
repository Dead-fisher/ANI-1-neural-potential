# ANI-1-neural-potential
ANI-1 neural potential for energy predicting

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
