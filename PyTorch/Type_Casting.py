#Type Casting with Pytorch
#Type Casting : �������� ����


import torch
import numpy as np

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float()) #float������ �����ؼ� ���
bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long()) #���������� �����ؼ� ���
print(bt.float()) #�Ǽ������� �����ؼ� ���