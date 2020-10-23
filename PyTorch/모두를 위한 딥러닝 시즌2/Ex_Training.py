import torch
from torch import optim

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

#모델 초기화
W = torch.zeros(1)


#Learning rate 설정
lr = 0.1

#Epoch: 데이터로 학습한 횟수
nb_epochs = 10
for epoch in range(nb_epochs+1):
    
    #H(x) 계산
    hypothesis = x_train * W
    
    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)
    
    print("Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}".format(
        epoch, nb_epochs, W.item(), cost.item()
    ))
    
    # cost gradient로 H(x) 개선
    W -= lr * gradient
print(cost)
