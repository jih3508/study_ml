#Sum with PyTorch

import torch

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t) 

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))
