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

#모델 초기화
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
    
    #표(x) 계산
    hypothesis = model(x_train) # H(x) → P(x=1) = 1- 1/ 1+e**(-W*x+b)
    
    #cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    #cost로 표(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) #각 데이터 참 or 거짓 판별
        correct_prediction = prediction.float() == y_train #boolean type을 float형으로 변환 해서 결과 값과 비교
        accurany = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accurany * 100,
        ))    