#Type Casting with Pytorch
#Type Casting : 데이터형 변경


import torch
import numpy as np

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float()) #float형으로 변경해서 출력
bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long()) #정수형으로 변경해서 출력
print(bt.float()) #실수형으로 변경해서 출력