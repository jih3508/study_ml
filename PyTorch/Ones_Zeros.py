#Ones and Zeros with Pytorch


import torch
import numpy as np

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x)) # 1로만 가득찬 똑같은 사이즈 Tensor
print(torch.zeros_like(x))  # 0으로만 가득찬 똑같은 사이즈 Tensor
