#In-place Operation with Pytorch

import torch
import numpy as np

x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.)) # r
print(x)
print(x.mul_(2.)) # x�� 2�� ���ؼ� x������ �־��
print(x)