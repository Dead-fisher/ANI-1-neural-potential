import torch
import torch.nn as nn
import AniDataProcesser as ADP
import os
import ANI_model as ani
import matplotlib.pyplot as plt
import utils as u

'''
# TODO
实现验证集的结果验证
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


def validate(AEV, ene, species):
    model.eval()
    with torch.no_grad():
        pred_val = model(AEV, species)
        loss_val, name = ani.exp_cost(pred_val, ene)
    print("Loss on validation data set:{:3F}. ".format(loss_val))
    return loss_val, name


# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = "./data/"
log_train = 'logging_train.txt'
log_val = 'logging_val.txt'
ani.init_log(log_train)
ani.init_log(log_val)
train_test_ratio = u.Train_Test_Ratio
atomic_number_dic = u.Atomic_Number_Dic

ad_loader = ADP.ani_data_loader(os.path.join(dataset_dir, "ani1_gdb10_ts.h5"))
AEV_transformer = ADP.AEV_transformer(atomic_number_differentiated=True)

# create model
model = ani.MoleculeANI(atoms=u.ATOM_CASES)

optimizer = torch.optim.AdamW(
        model.parameters(), lr=u.Learning_Rate, weight_decay=0.00005, betas=u.BETAS, eps=u.EPS
    )

best_validate = {'MSE': [], 'exponential': []}

epochs = 20
model.to(device)
model.train()
species = []
train_AEV, test_AEV, train_ene, test_ene = 0, 0, 0, 0

print('Start training ...')
for epoch in range(epochs):
    for i, data in enumerate(ad_loader):

        # Data Processing BELOW
        species = data['species']
        train_AEV, test_AEV, train_ene, test_ene = AEV_transformer(data)
        # Data Processing ABOVE

        pred = model(train_AEV.to(device), species)

        optimizer.zero_grad()
        loss, loss_name = ani.exp_cost(pred, train_ene.to(device))
        loss.backward()
        optimizer.step()

        # loss.to(torch.device('cpu'))
        ani.write_log(log_train, epoch, i, loss.item(), loss_name)

        if i % 1 == 0:
            print('Training for Epoch {:d} loop {:d}, {} loss is {:4F}'.format(epoch, i, loss_name, loss.item()))

    loss_val, loss_name = validate(test_AEV.to(device), test_ene.to(device), species)
    loss_val.to(torch.device('cpu'))
    ani.write_log(log_val, epoch, 1, loss_val.item(), loss_name)

    if len(best_validate[loss_name]) == 0:
        best_validate[loss_name].append(loss_val.item())
        model.save_model(optimizer, epoch, loss_name)

    elif loss_val.item() < best_validate[loss_name][-1]:
        best_validate[loss_name].append(loss_val.item())
        model.save_model(optimizer, epoch, loss_name)
        print("#-#-" * 30)
        print('At Epoch {}, model saved in ./models folder'.format(epoch))
        print('#  #'*40)

# TODO

ad_loader.cleanup()

