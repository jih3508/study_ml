#1D Array with Numpy

import numpy as np
import torch

t = np.array([0., 1., 2., 3., 4., 5., 6.,]) #1���� array ����
print(t)

print('Rank  of t: ', t.ndim)  # ������ �˷���
print('Size  of t: ', t.size)  # ��ü ������ �˷���
print('Shape of t: ', t.shape) # ����� �˷���

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element (0��° , 2��°, ��������° ���)
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1])   #Slicing  (2��° ���� 4��°����, 4��° ���� �������� ����)
print('t[:2] t[3:]     = ', t[:2], t[3:])      #Slicing  (1��° ����, 3��°���� ������)