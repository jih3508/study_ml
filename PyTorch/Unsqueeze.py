#Unqueeze with Pytorch
#unsqueeze(x) : x��ġ�� ���ο� ���� ����


import torch
import numpy as np

ft = torch.FloatTensor([0, 1, 2])
print(ft.shape)

print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.view(1, -1))
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1)) # dim = -1 ������ ����
print(ft.unsqueeze(-1).shape)


