import torch
import numpy as np
import AniDataProcesser as ADP
import os

dataset_dir = "./data/"
train_test_ratio = 0.8

atomic_number_dic = {
    'H': 1,
    'C': 6,
    'O': 8,
    'N': 7,
    'S': 16
}

ad_loader = ADP.ani_data_loader(os.path.join(dataset_dir, "ani1_gdb10_ts.h5"))
AEV_transformer = ADP.AEV_transformer(atomic_number_differentiated=True)

ele = []

for data in ad_loader:
    species = data['species']
    # print('S' in species)
    # print(species)
    for atom in species:
        if atom in ele:
            continue
        else:
            ele.append(atom)

with open('atoms.txt', 'w') as f:
    for atom in ele:
        f.write(atom)
        f.write('\n')


    # train_AEV, test_AEV, train_ene, test_ene = AEV_transformer(data)
    # print(train_AEV.shape, test_AEV.shape, train_ene.shape, test_ene.shape)
    #TODO


ad_loader.cleanup()

