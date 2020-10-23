#Mul vs Matmul with PyTorch
import torch

print('=============')
print('Mul vs Matmul')
print('=============')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 ¡¿ 2
print('Shape of Matrix 2: ', m2.shape) # 2 ¡¿ 1
print(m1.matmul(m2)) # 2 ¡¿ 1 [[1,2],[3,4]] ¡¿ [[1,2]]
print(m1 * m2) # 2 ¡¿ 2 [[1,2],[3,4]] ¡¿ [[1,1],[2,2]]
print(m1.mul(m2))