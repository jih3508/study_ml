import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# dataloader = DataLoader(
#                   dataset,
#                   batch_size=2  #�� minibatch�� ũ��(��������� 2�� �������� ����)
#                   shuffle = True) # Epoch ���� �����ͼ��� ���, �����Ͱ� �н��Ǵ� ������ �ٲ۴�.
                                    

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data =[[73, 80, 75],
                      [93, 88, 93],
                      [89, 91, 90],
                      [96, 98, 100],
                      [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self): # �����ͼ��� �� ������ ��
        return len(self.x_data)
    
    def __getitem__(self, idx): # ��� �ε��� idx�� �޾��� ��,
                                # �׿� �����ϴ� ����� ������ ��ȯ
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
               batch_size=2,  #�� minibatch�� ũ��(��������� 2�� �������� ����)
               shuffle = False) # Epoch ���� �����ͼ��� ���, �����Ͱ� �н��Ǵ� ������ �ٲ۴�.
    
#�� �ʱ�ȭ
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x) ���
        prediction = model(x_train)
        
        # cost ���
        cost = F.mse_loss(prediction, y_train)
        
        # cost�� H(x) ����
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # 20������ �α� ���
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:6f}'. format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
        ))
