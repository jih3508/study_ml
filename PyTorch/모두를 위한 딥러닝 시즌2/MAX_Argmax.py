#Max and Argmax with PyTorch
#Max: 가장 큰값
#Argmax: 가장 큰값의 인덱스값

import torch
import numpy as np

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())

print(t.max(dim = 0))
print('Max:    ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))
