import torch
from torch import optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

#������ ����
W = torch.zeros(1, requires_grad= True) # Weight �� Bias 0���� �ʱ�ȭ
b = torch.zeros(1, requires_grad= True) # requires_grad= True: �н��� ���̶�� ���

#hypothesis �ʱ�ȭ
hypothesis = x_train * W +b
cost = torch.mean((hypothesis - y_train) ** 2) #���� ��� ���

#Optimizer ���� 
optimizer = optim.SGD([W, b], lr=0.01) #[W,b]�� tensor
                                       #lr: learning rate
                                       
nb_epochs = 1000   
for epoch in range(1,nb_epochs+1):
    hypothesis = x_train * W +b                     #Hypothesis ����
    cost = torch.mean((hypothesis - y_train) ** 2)  #Cost ���  
    
    optimizer.zero_grad() #gradient �ʱ�ȭ           #Optimizer�� �н�
    cost.backward()       #gradient ���
    optimizer.step()      #����