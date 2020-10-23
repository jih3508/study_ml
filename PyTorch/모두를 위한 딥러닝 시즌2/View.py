#View (Reshape) with Pytorch
#View : tensor의 형태를 바꾸어 준다.

import torch
import numpy as np


t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9,10,11]]])

ft = torch.FloatTensor(t) # |ft|
print(ft.shape)

print(ft.view([-1, 3])) #-1: n차원은 신경안씀(변동이 심한곳에 사용) , 3: n차원을 3개로 묶어라
print(ft.view([-1, 2]).shape) # |ft| = (2,2,3) →(2 × 2, 3) = (4, 3)

print(ft.view([-1, 1, 3]))
print(ft.voew([-1, 1, 3]).shape)

