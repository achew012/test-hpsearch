#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


# ### Main Skeleton

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# #### Generates the most likely word from vocab

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ###  Utility Functions

# #### Function that clones blocks of encoders

# In[4]:


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# #### Initialize Embedding Parameters

# In[5]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ####                                    
# 
# $$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$
# 
# $$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$
# 
# where $pos$ is the position and $i$ is the dimension.  That is, each dimension of the positional encoding corresponds to a sinusoid.  The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.
# 

# In[6]:


# The positional encodings have the same dimension size (model_dim) as the embeddings
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        
        # Initialize the encodings
        pe = torch.zeros(max_len, d_model)
        # Word position
        position = torch.arange(0, max_len).unsqueeze(1)
        # In log space - exponent is reduced to multiplication & exponent of log is the original input
        # Denominator, 1/10000^(2i/model_dim) = -(2i/model_dim)*log(10000)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # Each dimension, i corresponds to a sinusoid - 2i is every 2 dimensions
        # sine encoding, PE(pos, 2i) = sin(pos/10000^(2i/model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        # cosine encoding, PE(pos, 2i+1) = cos(pos/10000^(2i/model_dim))
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# #### Implements
# $$                                                                         
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V               
# $$   

# In[7]:


# Multiply differentiable softmax-weighted query-key scores to value matrix
# Attention(Q:matrix,K:matrix,V:matrix) = softmax(Q(K_T)/sqrt(dimension of K))*V
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) #/ math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# #### Split attention score matrix to 8 heads

# #### 
# Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.                                            
# $$    
# \mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O    \\                                           
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)                                
# $$                                                                                                                 
# 
# Where the projections are parameter matrices $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.          

# In[8]:


# Split the attention score matrix (weighted query-key scores) into 8 independent parts
# Each of the independent part learn different attention weights on the value
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # why 4 linear layers?
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        
        # This mask is used if using the decoder part of transformers to prevent leftward flow of info by 
        # setting to -inf
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # batch size of input
        nbatches = query.size(0) 
        
        #Number of Heads * Linear(Concat(Attention(Linear(Value), Linear(Key), Linear(Q))))
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# #### 2 Linear transforms in the feedforward network
# 
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$    

# In[9]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ### Encoder

# #### Implements the layer normalisation from https://arxiv.org/abs/1512.03385

# In[10]:


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# #### Implements the "Add & Norm" component in transformers, sublayer output =  (input+normalized output)

# In[11]:


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# #### Defines One Encoder block/layer

# In[12]:

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 2 sublayers in an encoder block
        self.sublayer = clones(SublayerConnection(size, dropout), 2) 
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 1st "Add&Norm" sublayer applied to multi-head attention output
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        # 2nd "Add&Norm" sublayer applied to feedforward output
        return self.sublayer[1](x, self.feed_forward)


# #### Main Encoder class that arranges the encoder blocks/layers (clones) and normalization (layerNorm)

# class Encoder(nn.Module):
#     "Core encoder is a stack of N layers"
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         # last layer norm applied after N-encoder blocks
#         self.norm = LayerNorm(layer.size) 
        
#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, embedder, final_layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # last layer norm applied after N-encoder blocks
        self.norm = LayerNorm(layer.size)
        # embeddings with positional encoding
        self.embed = embedder
        self.final_layer = final_layer
        
    def forward(self, x, mask=None):
        x = self.embed(x)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return F.log_softmax(self.final_layer(self.norm(x)), dim=-1)

# ### Decoder

# In[14]:


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# In[15]:


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# #### Main Decoder class that arranges the encoder blocks/layers (clones) and normalization (layerNorm)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


def make_model(ntokens, N=6, 
               d_model=512, d_ff=1024, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    embedder = nn.Sequential(Embeddings(d_model, ntokens), c(position))
    final_layer = nn.Linear(d_model, ntokens)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N, embedder, final_layer)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

