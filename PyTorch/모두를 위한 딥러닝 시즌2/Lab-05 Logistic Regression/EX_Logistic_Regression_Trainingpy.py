import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # |x_data| (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] # |y_data| = (6,2)


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#모델 초기화
W = torch.zeros((2, 1), requires_grad= True) 
b = torch.zeros(1, requires_grad= True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    #Cost 개선
    hypothesis = torch.sigmoid(x_train.matmul(W) + b) # op .mn or @
    cost  = F.binary_cross_entropy(hypothesis, y_train)
    
    # cost로 표(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    #100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis[:5])

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction[:5])
print(y_train[:5])

correct_prediction = prediction.float() == y_train
print(correct_prediction[:5])