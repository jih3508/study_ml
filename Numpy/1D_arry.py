#1D Array with Numpy

import numpy as np
import torch

t = np.array([0., 1., 2., 3., 4., 5., 6.,]) #1차원 array 생성
print(t)

print('Rank  of t: ', t.ndim)  # 차원을 알려줌
print('Size  of t: ', t.size)  # 전체 사이즈 알려줌
print('Shape of t: ', t.shape) # 행려를 알려줌

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element (0번째 , 2번째, 마지막번째 출력)
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])   #Slicing  (2번째 부터 4번째까지, 4번째 부터 마직막전 까지)
print('t[:2] t[3:]     = ', t[:2], t[3:])      #Slicing  (1번째 까지, 3번째부터 끝까지)