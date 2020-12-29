import torch
import numpy as np
import AniDataProcesser as ADP
import os

dataset_dir = "../data/"
train_test_ratio = 0.8

atomic_number_dic = {
    'H': 1,
    'C': 6,
    'O': 8,
    'N': 7,
    'S': 32
}

ad_loader = ADP.ani_data_loader(os.path.join(dataset_dir, "ani1_gdb10_ts.h5"))
AEV_transformer = ADP.AEV_transformer(atomic_number_differentiated=True)

for data in ad_loader:
    # print(data)
    paths = data["path"]
    coordinates = data["coordinates"]
    # print('coord')
    # print(coordinates.shape)
    energies = data["energies"]
    species = data['species']
    # print('species')
    # print(np.array(species).shape)
    ass = (np.array(species) == 'C').astype(np.float)
    print(ass.sum())
    ass = (np.array(species) == 'H').astype(np.float)
    print(ass.sum())
    ass = (np.array(species) == 'O').astype(np.float)
    print(ass.sum())
    ass = (np.array(species) == 'N').astype(np.float)
    print(ass.sum())

    smiles = data['smiles']

    '''
    print("Path: ", paths)
    print(" Smiles: ", "".join(smiles))
    print(" Symbos: ", species)
    print(" Shape of coordinates: ", coordinates.shape)
    print(" Shape of energies: ", energies.shape)
    print(" Energies: ", energies, "\n")
    '''

    upper_bound = int(coordinates.shape[0]*train_test_ratio)
    train_crd = torch.tensor(coordinates[:upper_bound, :])
    test_crd = torch.tensor(coordinates[upper_bound:, :])
    train_ene = torch.tensor(energies[:upper_bound])
    test_ene = torch.tensor(energies[upper_bound:])
    
    atomic_number = torch.tensor([atomic_number_dic[i] for i in species])
    
    train_AEV = [AEV_transformer(train_crd[i], atomic_number) for i in range(upper_bound)]
    train_AEV = torch.stack(train_AEV)

    print(train_AEV.shape, test_ene.shape)
    #TODO
    
    

ad_loader.cleanup()

