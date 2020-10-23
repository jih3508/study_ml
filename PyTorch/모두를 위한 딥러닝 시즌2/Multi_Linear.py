import torch
from torch import optim

#데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#모델 초기화
w = torch.zeros((3, 1), requires_grad= True)
b = torch.zeros(1, requires_grad= True)

print(w)
# optimizer 설정
optimizer = optim.SGD([w, b], lr=1e-5)


nb_epochs = 20
for epoch in range(nb_epochs +1):
    
    #H(x) 계산 , matmul(w):로 한번에 계산
    hypothesis = x_train.matmul(w) + b # or .mm or @
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(),
        cost.item()
    ))