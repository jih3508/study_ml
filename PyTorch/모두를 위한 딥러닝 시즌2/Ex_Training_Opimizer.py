import torch
from torch import optim

# ������
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

#�� �ʱ�ȭ
W = torch.zeros(1, requires_grad=True)


#Learning rate ����
optimizer = optim.SGD([W], lr=0.15)

#Epoch: �����ͷ� �н��� Ƚ��
nb_epochs = 10
for epoch in range(nb_epochs+1):
    
    #H(x) ���
    hypothesis = x_train * W
    
    # cost gradient ���
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    print("Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}".format(
        epoch, nb_epochs, W.item(), cost.item()
    ))
    
    # cost�� H(x) ����
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
