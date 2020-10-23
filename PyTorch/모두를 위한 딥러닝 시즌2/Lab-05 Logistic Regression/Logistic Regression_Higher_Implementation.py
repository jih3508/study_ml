import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#�� �ʱ�ȭ
W = torch.zeros((8, 1), requires_grad= True) 
b = torch.zeros(1, requires_grad= True)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8,1) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1) 

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    
    #ǥ(x) ���
    hypothesis = model(x_train) # H(x) �� P(x=1) = 1- 1/ 1+e**(-W*x+b)
    
    #cost ���
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    #cost�� ǥ(x) ����
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20������ �α� ���
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) #�� ������ �� or ���� �Ǻ�
        correct_prediction = prediction.float() == y_train #boolean type�� float������ ��ȯ �ؼ� ��� ���� ��
        accurany = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accurany * 100,
        ))    