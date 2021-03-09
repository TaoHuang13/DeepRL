import torch

a = [torch.tensor([1,2,3]), torch.tensor([2,3,4]), torch.tensor([3,4,5])]
a = torch.stack(a)
print(a)
print(a.size())