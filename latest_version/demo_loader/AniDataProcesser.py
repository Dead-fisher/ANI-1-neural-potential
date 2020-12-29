#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    System Enviroment
    OS: MacOS Catalina 10.15.7
    CPU: Intel(R) Core(TM) i5 @ 2.3 GHz
    GPU: None
    Memory: 8 GiB

    Python Enviroment
    python==3.8.5
    numpy==1.19.1
    h5py==3.1.0
    pyTorch=1.6.0

    Author: Lin Xiaohan
    Email: linxiaohan@pku.edu.cn
    Ref: 
        https://github.com/isayev/ANI1_dataset 
            written by Roman Zubatyuk and Justin S. Smith
        https://github.com/deepchem/deepchem/blob/master/deepchem/trans/transformers.py
            written by 
    Latest Update: 2020.12.26
'''

import h5py
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class data_packer(object):
    def __init__(self, file_path, mode='w-', complib='gzip', complevel=6):
        '''
        Wrapper to store arrays within HFD5 file.
        
        Args:
            file_path: file path where .h5 file saved.
            mode: mode to open .h5 file, default "w-" means create file, fail if exists.
            complib: libarary used to compress the data.
            complevel: compression level.
        
        Returns:
            None.
        '''
        self.store = h5py.File(file_path, mode=mode)
        self.clib = complib
        self.clev = complevel

    def store_data(self, store_loc, **kwargs):
        '''
        Put arrays to store.

        Args:
            store_loc: location to store data.
        
        Returns:
            None.
        '''
        g = self.store.create_group(store_loc)
        for k, v, in kwargs.items():
            if (type(v) == list) and (len(v) != 0) \
                and (type(v[0]) is np.str_ or type(v[0]) is str):
                v = [a.encode('utf8') for a in v]
                
            g.create_dataset(k, data=v, compression=self.clib, compression_opts=self.clev)

    def clean_up(self):
        '''
        Close files.
        '''
        self.store.close()

class ani_data_loader(object):

    def __init__(self, file_path):
        '''
        Wrapper to load ANI data set.
        
        Args:
            file_path: file to store data files.
        
        Returns:    
            None.
        '''
        if not os.path.exists(file_path):
            exit('Error: file not found - '+file_path)
        self.store = h5py.File(file_path)

    def h5py_dataset_iterator(self, g, prefix=''):
        '''
        Group recursive iterator. Iterate through all groups in 
        all branches and return datasets in dicts.

        Args:
            g: group
            prefix: current work dictionary
        
        Returns:
            yield datasets iterator in python dict form
        '''
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):
                data = {'path':path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k])

                        if (type(dataset) is np.ndarray) and (dataset.size != 0)\
                            and (type(dataset[0]) is np.bytes_):
                            dataset = [a.decode('ascii') for a in dataset]

                        data.update({k:dataset})

                yield data
            else: # if item is group, then go down to unpack group into datasets
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        '''
        Default class iterator (iterate through all data).
        '''
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        '''
        Returns a list of all groups in the file.
        '''
        return [g for g in self.store.values()]

    def iter_group(self,g):
        '''
        Allows interation through the data in a given group.
        '''
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        '''
        Get the requested datasets.

        Args:
            path: the requested path for specific datasets.
        
        Returns:
            the requested datasets.
        '''
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k])

                if (type(dataset) is np.ndarray) and (dataset.size != 0)\
                    and (type(dataset[0]) is np.bytes_):
                    dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    def group_size(self):
        '''
        Returns the number of groups.
        '''
        return len(self.get_group_list())

    def size(self):
        '''
        Returns the number of items.
        '''
        count = 0
        for g in self.store.values():
            count += len(g.items())
        return count

    def cleanup(self):
        '''
        Close files.
        '''
        self.store.close()


class AEV_transformer(object):
    def __init__(self, max_atoms=23, radial_cutoff=4.6, angular_cutoff=3.1, radial_length=32,
                    angular_length=8, atom_cases=[1, 6, 7, 8, 16], atomic_number_differentiated=True,
                    coordinates_in_bohr=False):
        '''
        Convert coordinates into AEV vectors.

        Args:
            max_atoms: max atom numbers allowed.
            radial_cutoff: cutoff value for radial function.
            angular_cutoff: angular value for angular value.
            radial_length: numbers of radial symmetry function.
            angular_length: numbers of angular symmetry function
            atom_cases: possible atomic numbers.
            atomic_number_differentiated: whether using differentiated atomic number funtion or not
            coordinates_in_bohr: whether coordinates are given in Angtrom or bohr.
        
        Returns:
            None.
        '''
        self.max_atoms = max_atoms
        self.radial_cutoff = radial_cutoff
        self.angular_cutoff = angular_cutoff
        self.radial_length = radial_length
        self.angular_length = angular_length
        self.atom_cases = atom_cases
        self.atomic_number_differentiated = atomic_number_differentiated
        self.coordinates_in_bohr = coordinates_in_bohr
        
    def distance_matrix(self, coordinates):
        '''
        Calculate the distance matrix for all atoms
        
        Args:   
            coordinates: input coordinates [atom_numbers, 3].
        
        Returns:
            distance matrix in torch.tensor form [atom_numbers, atom_numbers].
        '''
        return torch.sqrt((-2*coordinates.mm(coordinates.t()))+\
                          torch.sum(torch.square(coordinates),axis=1,keepdim=True)+\
                          torch.sum(torch.square(coordinates.t()),axis=0,keepdim=True))
    
    def distance_cutoff(self, distance_matrix):
        '''
        Generate distance matrix with trainable cutoff.
        
        Args:
            distance_matrix: un-cutoffed distance matrix in torch.tensor form [atom_numbers, atom_numbers]
        
        Returns:
            cutoffed distance matrix in torch.tensor form [atom_numbers, atom_numbers]
        '''
        one = torch.ones_like(distance_matrix)
        zero = torch.zeros_like(distance_matrix)
        mask = torch.where(distance_matrix < self.radial_cutoff, one, zero)
        distance_matrix = 0.5 * (torch.cos(np.pi * distance_matrix / self.radial_cutoff)+1)
        return distance_matrix*mask-torch.eye(distance_matrix.shape[0])
    
    def radial_symmetry(self, d_cutoff, d, atomic_numbers):
        '''
        Generate radial symmetry function.
        
        Args:
            d_cutoff: cutoffed distance matrix in torch.tensor form [atom numbers, atom numbers].
            d: uncutoffed distance matrix in torch.tensor form [atom numbers, atom numbers].
            atomic_numbers: atomic numbers array for input coordinates [atom numbers, ].
            
        Returns:
            radial symmetry function array in torch.tensor form [atom_numbers, radial length].
        '''
        Rs = torch.linspace(0, self.radial_cutoff, self.radial_length).view(-1,1,1)
        eta = torch.ones_like(Rs) * 3 / (Rs[1] - Rs[0])**2
        d_cutoff = torch.stack([d_cutoff]*eta.shape[0])
        d = torch.stack([d]*eta.shape[0])
        
        out = torch.exp(-eta*torch.square(d - Rs)) * d_cutoff
        one = torch.ones_like(atomic_numbers)
        zero = torch.zeros_like(atomic_numbers)

        if (self.atomic_number_differentiated):
            out_tensors = []
            for atom_type in self.atom_cases:
                mask = torch.where(atomic_numbers == atom_type, one, zero)
                out_tensors.append(torch.sum(out*mask, axis=2))
            return torch.transpose(torch.cat(out_tensors), 1, 0)
                
        else:
            return torch.transpose(torch.sum(out, axis=2), 1, 0)
    
    def angular_symmetry(self, d_cutoff, d, atomic_numbers, coordinates):
        '''
        Generate angular symmetry functions.
        
        Args:
            d_cutoff: cutoffed distance matrix in torch.tensor form [atom numbers, atom numbers].
            d: uncutoffed distance matrix in torch.tensor form [atom numbers, atom numbers].
            atom_numbers: atomic numbers array for input coordinates [atom numbers, ].
            cooridnates: input coordinates [atom_numbers, 3].
        
        Returns:
            angular symmetry function in torch.tensor form [atom numbers, angular_length]
        '''
        
        Rs = torch.linspace(0, self.angular_cutoff, self.angular_length).view(-1,1,1,1)
        eta = float(3 / (Rs[1] - Rs[0])**2)
        thetas = torch.linspace(0, np.pi, self.angular_length).view(-1,1,1,1)
        zeta = float(self.angular_length**2)
        
        tot_atom_numbers = coordinates.shape[0]
        
        # calculate theta_ijk, shape[atom numbers, atom numbers, atom numbers]
        vector_distance = torch.stack([coordinates]*tot_atom_numbers, axis=0)\
                            -torch.stack([coordinates]*tot_atom_numbers, axis=1)
        R_ij = torch.stack([d]*tot_atom_numbers, axis=2)
        R_ik = torch.stack([d]*tot_atom_numbers, axis=1)
        f_R_ij = torch.stack([d_cutoff]*tot_atom_numbers, axis=2)
        f_R_ik = torch.stack([d_cutoff]*tot_atom_numbers, axis=1)
        
        vector_mul = torch.sum(torch.stack([vector_distance]*tot_atom_numbers, axis=2) * \
                              torch.stack([vector_distance]*tot_atom_numbers, axis=1), axis=3)
        vector_mul = vector_mul * torch.sign(f_R_ij) * torch.sign(f_R_ik)
        theta = torch.acos(vector_mul/(R_ij*R_ik+1e-5))
        
        out = torch.pow((1 + torch.cos(theta-thetas))/2, zeta) * \
                torch.exp(-eta  * torch.square((R_ij + R_ik) / 2 - Rs)) * f_R_ij * f_R_ik * 2
    
        iter_size = len(self.atom_cases)
        
        if (self.atomic_number_differentiated):
            out_tensors = []
            for iter_i in range(iter_size):
                for iter_j in range(iter_i+1, iter_size):
                    mask = torch.tensor([[1 if (int(atomic_numbers[i]) == self.atom_cases[iter_i] \
                                    and int(atomic_numbers[j]) == self.atom_cases[iter_j]) else 0\
                                   for i in range(tot_atom_numbers)] for j in range(tot_atom_numbers)])
                    out_tensors.append(torch.sum((out*mask), axis=[2,3]))
            return torch.transpose(torch.cat(out_tensors), 1, 0)
        
        else:
            return torch.transpose(torch.sum(out, axis=[2,3]), 1, 0)
    
    def __call__(self, input_crd, input_atomic_number):
        '''
        Transforms input coordinates and atomic numbers into AEV vectors.
        
        Args:
            input_crd: coordinates in torch.tensor form [atom numbers, 3]
            input_atomic_number: atomic numbers in torch.tensor form [atom numbers, 1]
                    
        Returns:
            AEV vector in the form of torch.tensor [atom_numbers, total AEV length]
        '''
        distance_matrix = self.distance_matrix(input_crd)
        distance_matrix_cutoff = self.distance_cutoff(distance_matrix)
        radial_symmetry_function = self.radial_symmetry(distance_matrix_cutoff, distance_matrix, input_atomic_number)
        angular_symmerty_function = self.angular_symmetry(distance_matrix_cutoff, distance_matrix, input_atomic_number, input_crd)
        return torch.cat([radial_symmetry_function, angular_symmerty_function], axis=1)
        