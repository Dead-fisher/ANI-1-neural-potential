import math

ATOM_CASES = ['C', 'H', 'O', 'N']
TAU = 0.5
Atomic_Number_Dic = {
    'H': 1,
    'C': 6,
    'O': 8,
    'N': 7,
    'S': 32
}
Train_Test_Ratio = 0.8

Learning_Rate = 0.001
BETAS = (0.9, 0.999)
EPS = 1e-8