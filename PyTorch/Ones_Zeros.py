#Ones and Zeros with Pytorch


import torch
import numpy as np

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x)) # 1�θ� ������ �Ȱ��� ������ Tensor
print(torch.zeros_like(x))  # 0���θ� ������ �Ȱ��� ������ Tensor
