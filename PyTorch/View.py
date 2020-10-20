#View (Reshape) with Pytorch
#View : tensor�� ���¸� �ٲپ� �ش�.

import torch
import numpy as np


t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9,10,11]]])

ft = torch.FloatTensor(t) # |ft|
print(ft.shape)

print(ft.view([-1, 3])) #-1: n������ �Ű�Ⱦ�(������ ���Ѱ��� ���) , 3: n������ 3���� �����
print(ft.view([-1, 2]).shape) # |ft| = (2,2,3) ��(2 �� 2, 3) = (4, 3)

print(ft.view([-1, 1, 3]))
print(ft.voew([-1, 1, 3]).shape)

