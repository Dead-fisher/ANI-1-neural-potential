#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    System Enviroment
    OS: MacOS Catalina 10.15.4
    CPU: 1.6 GHz Dual-Core Intel Core i5
    GPU: None
    Memory: 8 GB 2133 MHz LPDDR3

    Python Enviroment
    python==3.7.6
    numpy==1.18.1
    h5py==2.10.0
    pyTorch==1.6.0

    Author: Wang Yanze
    Email: wyze@pku.edu.cn
    Ref:
        https://github.com/aiqm/torchani/tree/master/torchani
        written by Ignacio Pickering
        https://github.com/isayev/ASE_ANI
        written by Olexandr Isayev
    cite:
    Xiang Gao, Farhad Ramezanghorbani, Olexandr Isayev, Justin S. Smith, and Adrian E. Roitberg.
    TorchANI: A Free and Open Source PyTorch Based Deep Learning Implementation of the ANI Neural Network Potentials.
    Journal of Chemical Information and Modeling 2020 60 (7), 3408-3415,

    Latest Update: 2020.12.28
"""


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def initial_para(module):
    '''
    Initialization of parameters of module.
    Weights of linear layer will be set randomly based on normal distribution.
    Bias of linear layer will be set to be 0.

    Returns:
        None.
    '''
    if type(module) == nn.Linear:
        torch.nn.init.kaiming_normal_(module.weight, a=1.0)
        nn.init.zeros_(module.bias)
    return


class MoleculeANI(nn.Module):
    def __init__(self, atoms=None, species=None):
        """
        Customized nn.Module for ANI-1 net. A individual net will be constructed for
        every atom case in molecule data set.

        Args:
            atoms: Types of atom used for net construction. -> list/array
            species: all atoms that a molecule contains. -> list/array
        Returns:
            None.
        """
        super(MoleculeANI, self).__init__()
        if atoms is None:
            atoms = []
        self.species = species
        self.atoms = atoms
        # check whether atoms in species are in atoms
        if species is not None:
            self.check_atom_type()
        self.models = nn.ModuleDict({})  # Module Container for every atom type.
        # construct nets for every atom in atoms.
        for atom in atoms:
            self.models[atom] = self.create_atom_module()

        # initializing
        self.models.apply(initial_para)

    def forward(self, data, species):
        """
        Forward function(abstract method).

        Args:
            data: train/unknown data -> tensor with shape of [batch_size, atom number, features]
            species: all atoms that a molecule contains (in order). -> list/array
        Returns:
            Output(predicted energy).
        """
        if type(species) is not np.ndarray:
            species = np.array(species)
        species = species.flatten()
        # check for nan in data.
        data = torch.where(torch.isnan(data), torch.full_like(data, 0), data)

        # Initialization of output.
        output = torch.zeros((1, data.shape[0])).flatten()
        for i, atom in enumerate(self.atoms):
            # create mask for atom. True(or 1) represents such atom locates at this position.
            mask = (species == atom)
            # idx is the index where such atom locates in species list/array.
            # We will ues idx to slice the data w.r.t. the atom type.
            idx = mask.nonzero()[0]
            if idx.shape[0] > 0:  # if True means that at least one atom EXIST in this molecule.
                output += self.models[atom](data[:, idx, :]).sum(dim=1).flatten()
                # ultimate shape of output is [BatchSize, 1].
        return output

    @staticmethod
    def create_atom_module():
        '''
        create individual net for every atom.

        Returns:
            a module sequential.
        '''
        return nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.CELU(0.1),
            torch.nn.Linear(64, 1)
        )

    def check_atom_type(self):
        '''
        check whether atoms in species are also in atoms.

        Returns:
            None
        '''
        if self.atoms is None:
            self.atoms = []
        for i in self.species:
            if i not in self.atoms:
                self.atoms.append(str(i))

    def check_species(self, species):
        '''
        check whether atoms in NEW species are also in atoms.

        Returns:
            None
        '''
        new_atom = np.setdiff1d(species, self.atoms)
        if new_atom.shape[0] != 0:
            for atom in new_atom:
                self.models[atom] = self.create_atom_module()
        return


def check_species(species, atoms):
    '''
    check whether atoms in NEW species are also in atoms.

    Returns:
        intersection of the two lists/arrays.
    '''
    new_atom = np.setdiff1d(species, atoms)
    return new_atom


def exp_cost(output, label):
    '''
    Define loss function according to Chem. Sci., 2017, 8, 3192–3203 as:
    Loss = tau * exp( Sum (out-pred) ** 2 / tau)
    but it will be INF due to huge number. so a Piecewise function is defined.
    Args:
        output: predicted output
        label: label of data
    :return: loss
    '''
    loss = F.mse_loss(output, label, reduction='sum')
    if loss > 10:
        return loss, 'MSE'
    else:
        loss = 0.5 * torch.exp(2. * loss)
        return loss, 'exponential'


if __name__ == '__main__':
    my_ani = MoleculeANI(atoms=['H'])

    print(my_ani)
    '''bb = my_ani.named_parameters()
    for i in bb:
        print(i)'''
    '''print('children')
    print(my_ani.children())
    for i in my_ani.children():
        print(i)
    print('modules')
    for j in my_ani.modules():
        print(j)'''

    '''pp = my_ani.parameters()
    for i in pp:
        print(i)'''

    pass



