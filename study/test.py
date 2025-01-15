import torch

a = torch.tensor(
    [[0.2035, 1.2959, 1.8101, -0.4644],
     [1.5027, -0.3270, 0.5905, 0.6538],
     [-1.5745, 1.3330, -0.5596, -0.6548],
     [0.1264, -0.5080, 1.6420, 0.1992]])

variance = torch.var(a, dim=1, unbiased=False, keepdim=False)
print(variance)

print(torch.mean(variance, dim=0))
