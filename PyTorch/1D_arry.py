#1D Array with PyTorch
import torch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.]) # Float형 Tensor를 만든다.
print(t)

print(t.dim())  #차원
print(t.shape)  #행렬
print(t.size()) #전체 크기
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])