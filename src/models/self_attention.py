# coding=utf-8
"""
# may be helpful :
Some tricks (especially for multi-head attention and positional encoding) and decoding Language Models
https://atcold.github.io/NYU-DLSP20/en/week12/12-1/#:~:text=maximum%20sequence%20length-,Some%20tricks%20(especially%20for%20multi%2Dhead%20attention%20and%20positional%20encoding)%20and,Really%20helpful%20for%20a%20task%20like%20machine%20translation,-The%20following%20are
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""
SelfAttention v.s. Attention classes v.s. FVT Attention:

Input shape:
    SelfAttention:(batch_size, max_len, dim) for queries, keys, and values.
    
    Attention (Cross attention ): /"query/" is a single vector (batch_size, key_dim),
               keys and values are sequences (batch_size, max_len, dim).

Masking:
    Both classes use masks to ignore certain positions in the sequence (e.g., padding tokens).
    Masks are applied to keys and values to set specific elements to 0.

Weight Computation:
    SelfAttention: Computes a full attention weight matrix (batch_size, max_len, max_len)
    ,where each token attends to all others in the sequence.
    
    Attention: Computes a simpler attention weight matrix (batch_size, 1, max_len)
    ,where the single query attends to the entire sequence.

Output:
    SelfAttention: Produces contextualized embeddings for all tokens in the sequence.
    SA Output Dim: (batch_size, max_len, value_dim) (64 , 6 , 512)
    Attention: Produces a single contextualized embedding for the input query.
    Att Output Dim: (batch_size, value_dim).

"""


"""
usage of the function :
# Usage
device = torch.device('cpu')
self_attention = SelfAttention(key_dim, value_dim, device)
output = self_attention(queries, keys, values, mask)

print("SelfAttention Output Shape:", output.shape)
# SelfAttention Output Shape: torch.Size([2, 4, 5])

"""
class SelfAttention(nn.Module):
    """key_dim will be used for Query_dim also  : Q.K^T """
    
    """if you used .byte() for the masks there will be error in masked_fill_ 
       you have to use .bool() instead .
       the dafault for the mask is .byte() ,
       so you must add .bool(). """
    def __init__(self, key_dim, value_dim, device):
        super(SelfAttention, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1) #(batch_size ,rows,cols) for key matrix for example
        
    def forward(self, queries, keys, values, mask):
        """
        Attention mechanisms use this mask to prevent 
        padded positions or irrelevant parts of the 
        sequence from affecting the computation of
        attention weights.
        """

        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim).bool() # (batch_size, max_len, key_dim)
        # query_mask = mask.unsqueeze(-1).repeat(1, 1, self.query_dim).bool()  # (batch_size, max_len, key_dim) 
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim).bool() # (batch_size, max_len, value_dim)
        
        queries.data.masked_fill_(key_mask.bool(), 0.)
        keys.data.masked_fill_(key_mask.bool(), 0.)
        values.data.masked_fill_(value_mask.bool(), 0.)

        max_len = keys.shape[1]
        #weights = queries @ keys.transpose(-2,-1)
        weights = torch.bmm(queries, keys.permute(0,2,1)) # (batch_size, max_len, max_len)

        #weight_mask = (weights == 0.0).byte() 
        weight_mask = torch.where(weights == 0.,
                                  torch.ones_like(weights),
                                  torch.zeros_like(weights)).bool()
        weights.data.masked_fill_(weight_mask.bool(), -float('inf'))
        

        # Apply softmax and handle NaNs
        # make the diagonal also infinit 
        for i in range(max_len):
            # for all batches make the row i and col i == inf
            weights[:,i,i] = -float('inf')

        weights = self.softmax(weights)
        # weight_mask = torch.isnan(weights) [matrix of True's and False's]
        weight_mask = torch.where(torch.isnan(weights),
                                  torch.ones_like(weights),
                                  torch.zeros_like(weights)).bool()
        weights.data.masked_fill_(weight_mask.bool(), 0.)
        
        # bmm as matmul but bmm is efficient for batch matrix-matrix multiplication
        return torch.bmm(weights, values) # (batch_size, max_len, value_dim) (64 , 6 , 512)