import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# dataloader = DataLoader(
#                   dataset,
#                   batch_size=2  #각 minibatch의 크기(통상적으로 2의 제곱수로 설정)
#                   shuffle = True) # Epoch 마다 데이터셋을 섞어서, 데이터가 학습되는 순서를 바꾼다.
                                    

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data =[[73, 80, 75],
                      [93, 88, 93],
                      [89, 91, 90],
                      [96, 98, 100],
                      [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self): # 데이터셋의 총 데이터 수
        return len(self.x_data)
    
    def __getitem__(self, idx): # 어떠한 인덱스 idx를 받았을 때,
                                # 그에 상응하는 입출력 데이터 반환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x,y
    
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(1)
x_data =[[73, 80, 75],
         [93, 88, 93],
         [89, 91, 90],
         [96, 98, 100],
         [73, 66, 70]]
y_data = [[152], [185], [180], [196], [142]]

x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)

dataloader = DataLoader(
               CustomDataset(),
               batch_size=2,  #각 minibatch의 크기(통상적으로 2의 제곱수로 설정)
               shuffle = False) # Epoch 마다 데이터셋을 섞어서, 데이터가 학습되는 순서를 바꾼다.
    
#모델 초기화
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)
        
        # cost 계신
        cost = F.mse_loss(prediction, y_train)
        
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # 20번마다 로그 출력
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:6f}'. format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
        ))
