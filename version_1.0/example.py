import torch
import torch.nn as nn
import numpy as np
import AniDataProcesser as ADP
import os
import ANI_model as ani
import matplotlib.pyplot as plt
import utils as u
import pandas as pd

'''
# TODO
实现验证集的结果验证
实现分布在cuda上训练，注意数据的储存位置。
模型的储存和下载，epoch后，如果验证集loss变小，就储存模型
完善模型储存路径
建立保存机制

# 实现结果的对比
实现结果分析
封装部分函数功能

# 部分函数实现与文献有出入
Loss函数，exp部分会上溢。
权重递减部分，只定义了一个总权重
正则化部分，采用了L2正则化，但原文献是max norm 正则化，无法直接在pytorch实现，需要手动写。

'''

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = "./data/"
train_test_ratio = u.Train_Test_Ratio
atomic_number_dic = u.Atomic_Number_Dic

ad_loader = ADP.ani_data_loader(os.path.join(dataset_dir, "ani1_gdb10_ts.h5"))
AEV_transformer = ADP.AEV_transformer(atomic_number_differentiated=True)
model = ani.MoleculeANI(atoms=u.ATOM_CASES)

optimizer = torch.optim.AdamW(
        model.parameters(), lr=u.Learning_Rate, weight_decay=0.00005, betas=u.BETAS, eps=u.EPS
    )
# TODO

loss_collect = []
step = []

best_validate = []

epochs = 10
model.train()

for epoch in range(epochs):
    for i, data in enumerate(ad_loader):

        # Data Processing BELOW
        paths = data["path"]
        coordinates = data["coordinates"]
        energies = data["energies"]
        species = data['species']
        smiles = data['smiles']

        upper_bound = int(coordinates.shape[0] * train_test_ratio)
        train_crd = torch.tensor(coordinates[:upper_bound, :])
        test_crd = torch.tensor(coordinates[upper_bound:, :])
        train_ene = torch.tensor(energies[:upper_bound], dtype=torch.float)
        test_ene = torch.tensor(energies[upper_bound:])

        atomic_number = torch.tensor([atomic_number_dic[i] for i in species])

        train_AEV = [AEV_transformer(train_crd[i], atomic_number) for i in range(upper_bound)]
        train_AEV = torch.stack(train_AEV)
        # Data Processing ABOVE

        optimizer.zero_grad()
        pred = model(train_AEV, species)
        # loss = torch.exp(criterion(pred, train_ene) / tau) * tau

        loss, loss_name = ani.exp_cost(pred, train_ene)
        loss_collect.append(loss)
        loss.backward()
        step.append(i)
        optimizer.step()

        if i % 10 == 0:
            print('Training for Epoch {:d} loop {:d}, {} loss is {:4F}'.format(epoch, i, loss_name, loss))


model = model.to(torch.device('cpu'))
state = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epochs': epoch
        }
if not os.path.isdir('./models'):
    os.mkdir('./models')
torch.save(state, './models/ani_para.pth')
print('model saved in ./models folder')
# TODO

plt.figure()
plt.plot(step, loss_collect)
plt.show()

ad_loader.cleanup()

