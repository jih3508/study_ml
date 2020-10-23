import torch
from torch import optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

#데이터 정의
W = torch.zeros(1, requires_grad= True) # Weight 와 Bias 0으로 초기화
b = torch.zeros(1, requires_grad= True) # requires_grad= True: 학습할 것이라고 명시

#hypothesis 초기화
hypothesis = x_train * W +b
cost = torch.mean((hypothesis - y_train) ** 2) #오차 평균 계산

#Optimizer 정의 
optimizer = optim.SGD([W, b], lr=0.01) #[W,b]는 tensor
                                       #lr: learning rate
                                       
nb_epochs = 1000   
for epoch in range(1,nb_epochs+1):
    hypothesis = x_train * W +b                     #Hypothesis 예측
    cost = torch.mean((hypothesis - y_train) ** 2)  #Cost 계산  
    
    optimizer.zero_grad() #gradient 초기화           #Optimizer로 학습
    cost.backward()       #gradient 계산
    optimizer.step()      #개선