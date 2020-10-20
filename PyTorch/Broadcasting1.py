#Broadcasting with PyTorch
import torch

# Same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] ¡æ [[3, 3]]
print(m1 + m2) # [[1,2]] + [[3,3]]

# 2 ¡¿ 1 Vector + 1 ¡¿ 2 Vector
m1 = torch.FloatTensor([[1, 2]])  # (1,2) ¡æ (2,2), [[1, 2]] ¡æ [[1, 2], [1, 2]]
m2 = torch.FloatTensor([[3],[4]]) # (2,1) ¡æ (2,2), [[3], [4]] ¡æ [[3, 3],[4, 4]]
print(m1 + m2) # [[1, 2], [1, 2]] + [[3, 3],[4, 4]]