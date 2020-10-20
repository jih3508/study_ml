#Mean with PyTorch

import torch

t = torch.FloatTensor([1, 2])
print(t.mean()) #평균값

#Can't use mean() on integers
t = torch.LongTensor([1,2]) #정수값
try:
    print(t.mean())
except Exception as exc:
    print(exc)
    
t = torch.FloatTensor([[1,2], [3, 4]])
print(t)
print(t.mean())
print(t.mean(dim = 0))  # Vector로 표시(세로 평균)
print(t.mean(dim = 1))  # Matrix로 표시 (가로 평균)
print(t.mean(dim = -1)) # Matrix로 표시 (가로 평균)
