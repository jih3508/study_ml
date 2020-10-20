#Squeeze with Pytorch
#Squeeze : 원소가 1인 차원을 제거
#Squeeze(dim = n): n차원을 제거

import torch
import numpy as np

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

