#1D Array with PyTorch
import torch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]]) # Float�� Tensor�� �����.
print(t)

print(t.dim())  #����
print(t.shape)  #���
print(t.size()) #��ü ũ��
print(t[0], t[1], t[-1]) #Element
print(t[2:5], t[4:-1])   #Slice
print(t[:2], t[3:])      #Slice