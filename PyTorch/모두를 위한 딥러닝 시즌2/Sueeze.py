#Squeeze with Pytorch
#Squeeze : ���Ұ� 1�� ������ ����
#Squeeze(dim = n): n������ ����

import torch
import numpy as np

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

