#Concatenate with Pytorch
#Concatenate : �� Tensor�� �̾� ���δ�.


import torch
import numpy as np

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0)) #0�������� �پ�� (4,2)
print(torch.cat([x, y], dim=1)) # 1�������� �پ�� (2,4)