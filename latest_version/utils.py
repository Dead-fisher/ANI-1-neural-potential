import math

"""
This file contains parameters, constants and some training initialization settings.
"""

ATOM_CASES = ['C', 'H', 'O', 'N']  # for types of nets.

TAU = 0.5  # for loss function.

Atomic_Number_Dic = {  # atom mass
    'H': 1,
    'C': 6,
    'O': 8,
    'N': 7,
    'S': 32
}
Train_Test_Ratio = 0.85  # for batch_size

Learning_Rate = 0.001

BETAS = (0.9, 0.999)  # for AdamW optimizer.
EPS = 1e-8