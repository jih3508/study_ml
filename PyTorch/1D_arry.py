#1D Array with PyTorch
import torch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.]) # Float�� Tensor�� �����.
print(t)

print(t.dim())  #����
print(t.shape)  #���
print(t.size()) #��ü ũ��
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])