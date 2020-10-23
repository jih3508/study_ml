#Stacking with Pytorch
#Stacking : °¢ TensorÀ» ½×´Ù.

import torch
import numpy as np

x = torch.FloatTensor([1, 2])
y = torch.FloatTensor([3, 4])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z])) 
print(torch.stack([x, y, z], dim=1)) 

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim = 0))