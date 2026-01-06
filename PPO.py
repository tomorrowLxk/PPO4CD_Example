import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_statu = torch.nn.Sequential(
            torch.nn.Linear(16, 128),
            nn.ReLU()
        )
        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )
        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        )


    def forward(self, state):
        # 检查线性层输出
        linear_output = self.fc_statu(state)
        mu = self.fc_mu(linear_output) * 2.0  # 动作值应该在-2到2之间
        std = self.fc_std(linear_output) + 1e-3  # 确保标准差有一个最小值
        return mu, std
