import torch

from torch import nn


class Resnet18(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

        self.fc1 = torch.nn.Linear(in_features=512, out_features=256)
        self.drop1 = torch.nn.Dropout(p=.2)
        self.out = torch.nn.Linear(in_features=256, out_features=68*2)

        
    def forward(self, inp):
        backbone_activation = self.backbone.forward(inp).squeeze()
        out = self.fc1(backbone_activation)
        out = self.drop1(out)
        out = self.out(out)
        out = torch.sigmoid(out)

        return out