import torch

a = [1,2,3,4]
a = torch.tensor(a).view(1,-1)
print(a.size())