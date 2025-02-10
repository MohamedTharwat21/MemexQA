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



# key - value (KV) attention 
"""single-query attention, where a
   single query (often from an external source) 
   attends to a sequence of keys and values."""
"""or Cross attention , 
    Example: Vision-Language Models
    In vision-language tasks like Visual Question Answering (VQA):
    Query: Encodes the question text (from a transformer-based language model).
    Key-Value Pairs: Represent image features (from a vision model like CNN or ViT).
    Cross-attention aligns the query (text) with the relevant parts of the image
    features to answer the question."""

class Attention(nn.Module):
    def __init__(self, key_dim, value_dim, device):
        super(Attention, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, keys, values, mask):
        """the Query comming from different source"""
        """Query if it was a single word representation [in the target translation for ex],
           i need to attend it to the other words in the [source] sentence."""
        
        # query won't be masked 'it's a vector not matrix'.
        # it will be used as it is. 
        # Expands query to match the sequence dimension for compatibility with keys.
        query = query.unsqueeze(1) # (batch_size, 1, key_dim) (bs,1,512)

        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim) # (batch_size, max_len, key_dim)
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim) # (batch_size, max_len, value_dim)
        
        keys.data.masked_fill_(key_mask.bool(), 0.)
        values.data.masked_fill_(value_mask.bool(), 0.)

        max_len = keys.shape[1]

        # Computes attention weights for the single query against all tokens in keys.
        weights = torch.bmm(query, keys.permute(0,2,1)) # (batch_size, 1, max_len)
        
        weight_mask = torch.where(weights == 0., torch.ones_like(weights), torch.zeros_like(weights)) 
        weights.data.masked_fill_(weight_mask.bool(), -float('inf'))
        
        #ignore diagonal infinity 
        
        weights = self.softmax(weights)
        weight_mask = torch.where(torch.isnan(weights), torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.bool(), 0.) # (batch_size, 1, max_len)
        
        return torch.bmm(weights, values).squeeze(1) # (batch_size, value_dim)
        # Attention Output Shape: torch.Size([2, 5])