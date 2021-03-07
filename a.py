import torch

a = torch.ones((64,1))
for i in range(64):
    a[i] = 2

print(a[2])