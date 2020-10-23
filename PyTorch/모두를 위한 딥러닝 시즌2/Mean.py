#Mean with PyTorch

import torch

t = torch.FloatTensor([1, 2])
print(t.mean()) #��հ�

#Can't use mean() on integers
t = torch.LongTensor([1,2]) #������
try:
    print(t.mean())
except Exception as exc:
    print(exc)
    
t = torch.FloatTensor([[1,2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim = 0))  # Vector�� ǥ��(���� ���)
print(t.mean(dim = 1))  # Matrix�� ǥ�� (���� ���)
print(t.mean(dim = -1)) # Matrix�� ǥ�� (���� ���)
