import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,dim=1,n=64, sigmoid=nn.GELU()):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(dim, n)
        self.fc_2 = nn.Linear(n, 1)
        self.sigmoid = sigmoid
        self.dim = dim

    def forward(self, x, y=None):
        if self.dim == 1:
            out = self.sigmoid(self.fc_1(x))
            out = self.fc_2(out)
            return out
        if self.dim == 2:
            input=torch.stack([x,y],dim=-1)
            out = self.sigmoid(self.fc_1(input))
            out = self.fc_2(out)
            return out
        else:
            raise ValueError("Only dim=1 or dim=2 is supported.")

if __name__ == '__main__':
    model = MLP()
    x = torch.Tensor([[1.0]])
    print(model(x))