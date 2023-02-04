from torch import nn
import torch.nn.functional as F

class CNNCifar(nn.Module):
    def __init__(self, args, hidden_dim=120):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, args.num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc0(x))
        x1 = F.relu(self.fc1(x))
        x = self.fc2(x1)
        return F.log_softmax(x, dim=1), x1


class MLP_Mnist(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_Mnist, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.layer_hidden1 = nn.Linear(512, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, 64)
        self.layer_out = nn.Linear(64, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x1 = self.layer_hidden2(x)
        x = self.relu(x1)
        x = self.layer_out(x)
        return self.softmax(x), x1



