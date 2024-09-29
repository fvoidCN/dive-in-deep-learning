import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
features = features
labels = labels


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))
print(net[0].weight.data.normal_(0, 0.01))
print(net[0].bias.data.fill_(0))

loss = nn.HuberLoss()

for p in net.parameters():
    print(p)

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
b = net[0].bias.data

print(true_w, true_b)
print(w, b)