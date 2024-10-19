import torch
from torch import nn as nn
from labml_helpers.module import Module

class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        '''
d_model 是标记嵌入中的特征数量
d_ff 是 FFN 隐藏层中的特征数量
dropout 是隐藏层的 Dropout 率
is_gated 指定了隐藏层是否为门控层
bias1 指定了第一个全连接层是否应该具有可学习的偏置
bias2 指定第二个全连接层是否应具有可学习的偏置
bias_gate 指定门控的全连接层是否应具有可学习的偏置
        '''
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)
    
    def forward(self,x:torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)                
        





























