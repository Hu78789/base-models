import math
from typing import Optional
import torch
import torch.nn as nn
from labml_nn.utils import clone_module_list
from .FFN import FeedForward
from .mha import MultiHeadAttention
from .positional_encoding import get_positional_encoding

class EmbeddingsWithPositionalEncoding(nn.Module):
    '''
    嵌入 token 并添加固定位置编码
    '''
    def __init__(self,d_model:int,n_vocab:int,max_len:int = 5000) -> None:
        super().__init__()
        self.linear = nn.Embedding(n_vocab,d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings',get_positional_encoding(d_model,max_len))
    
    def forward(self,x:torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].required_grad_(False)
        return self.linear(x)*math.sqrt(self.d_model) + pe    

class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    '''
    嵌入 token 并添加参数化的位置编码
    '''
    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len,1,d_model),requires_grad=True)
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe

class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: Optional[MultiHeadAttention] = None,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `src_attn` is the source attention module (when this is used in a decoder)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False
    def forward(self,*,x:torch.Tensor,
                mask:torch.Tensor,
                src:torch.Tensor|None = None,
                src_mask:torch.Tensor|None = None):
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z,value=z,key=z,mask=mask)
        x = x + self.dropout(self_attn)
        if src is not None:
            z = self.norm_src_attn(x)
            if self.src_attn is not None:
                attn_src = self.src_attn(query=z,key=src,value=src,mask=src_mask)      
                x = x + self.dropout(attn_src)
        z = self.norm_ff(x)
        if self.is_save_ff_input:
            self.ff_input = z.clone
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self,layer:TransformerLayer,n_layers:int):
        super().__init__()
        self.layers = clone_module_list(layer,n_layers)
        self.norm = nn.LayerNorm([layer.size])
    
    def forward(self,x:torch.Tensor,mask:torch.Tensor):
        for layer in self.layers:
            x = layer(x=x,mask=mask)
        return self.norm(x)
            
class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, n_vocab:int,d_model:int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model,n_vocab)
    def forward(self,x):
        return self.projection(x)

class EncoderDecoder(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:nn.Module,
                 tgt_embed:nn.Module,generator:nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,src:torch.Tensor,tgt:torch.Tensor,
                src_mask:torch.Tensor,tgt_mask:torch.Tensor):
        def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
            return self.encoder(self.src_embed(src), src_mask)

        def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
            return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)            
        
        enc = self.encoder(src,src_mask)
        return self.decode(enc,src_mask,tgt,tgt_mask)    

        
        
            
    
        
    
    
            
                            
                