import math
from typing import Optional,List

import torch
from torch import nn

from labml import tracker

class PrepareForMultiHeadAttention(nn.Module):
    '''
    该部分执行线性变换，并将向量分割成给定数量的头以获得多头注意力。这用于键、查询和值向量。
    '''
    def __init__(self,d_model:int,heads:int,d_k:int,bias:bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=d_model,out_features=heads*d_k)
        self.heads = heads
        self.d_k = d_k
    def forward(self,x:torch.Tensor):
        #输入的形状为[seq_len, batch_size, d_model] 或[batch_size, d_model] 。我们对最后一维应用线性变换，并将其分为多个头。
        #输出具有形状[seq_len, batch_size, heads, d_k] 或[batch_size, heads, d_model]
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape,self.heads,self.d_k)
        return x
    
class MultiHeadAttention(nn.Module):    
    def __init__(self,heads:int,d_model:int,dropout_prob:float = 0.1,bias:bool = True):
        super().__init__()
        #d_k特征数量
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model,heads,self.d_k,bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None
    def get_scores(self,query:torch.Tensor,key:torch.Tensor):
        return torch.einsum("ibhd,jbhd->ijbh",query,key)
    def prepare_mask(self,mask:torch.Tensor,query_shape:List[int],key_shape:List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]         
        mask = mask.unsqueeze(-1)
        return mask
    
    def forward(self,*,query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask:Optional[torch.Tensor] = None):
        seq_len,batch_size,_ = query.shape
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query,key)
        scores *= self.scale
        
        if mask is not None:
            mask = self.prepare_mask(mask,query.shape,key.shape) # type: ignore
        
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))
        attn = self.softmax(scores)
        #tracker.debug('attn',attn)
        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd",attn,value)    
    

        
            