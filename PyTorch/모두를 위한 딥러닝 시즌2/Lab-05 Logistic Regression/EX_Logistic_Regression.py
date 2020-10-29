#로지스틱 회귀: 참거짓을 알아내기 위한 분석방법 함수는 sigmoid 함수 사용
#sigmoid 0~1사이에서 S자 형태의 그래프

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # |x_data| (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] # |y_data| = (6,2)


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)


print('e^1 equals: ', torch.exp(torch.FloatTensor([1])))

W = torch.zeros((2, 1), requires_grad= True) 
b = torch.zeros(1, requires_grad= True)

#H(X) = 1/ 1+e**(-W*x+b)
hypothesis = 1 /(1 + torch.exp(-(x_train.matmul(W) + b)))
print(hypothesis)
print(hypothesis.shape)

print('1/(1+e^{-1}) equals: ', torch.sigmoid(torch.FloatTensor([1])))

hypothesis = torch.sigmoid(x_train.matmul(W) + b) #torch에서 sigmoid 제공함
                                                  #sigmoid(x) = e^x

print(hypothesis)
print(hypothesis.shape)

losses =-(y_train[0] * torch.log(hypothesis[0]) + 
          (1 - y_train[0]) * torch.log(1 - hypothesis))

print(losses)

cost = losses.mean()
print(cost)

#F.binary_cross_entropy(hypothesis, y_train): 손실값 평균 구해주는 함수

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis[:5])