#Concatenate with Pytorch
#Concatenate : 각 Tensor를 이어 붙인다.


import torch
import numpy as np

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0)) #0차원끼리 붙어라 (4,2)
print(torch.cat([x, y], dim=1)) # 1차원끼리 붙어라 (2,4)