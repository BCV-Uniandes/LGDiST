import torch.nn as nn
from utils import *
import torch
import math


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, feedforward_dim, latent_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.encoder_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh())

        self.positional_encoding = PositionalEncoding(embedding_dim)
        #self.positional_encoding = nn.Parameter(torch.randn(1, 7, embedding_dim))
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward= feedforward_dim,
            dropout=dropout,
        )
        
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.to_latent = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, latent_dim),
            nn.Tanh(),
        )
        
    
    def forward(self, x):
        # Input shape: [Batch, 7, 1024]
        #breakpoint()
        x = self.encoder_projection(x) # Shape: [Batch, 7, Embedding Dim]
        x = self.positional_encoding(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x)  # Shape: [Batch, 7, Embedding Dim]
        x = x.permute(1, 0, 2)
        
        x = self.to_latent(x)  # Shape: [Batch, 7, Latent Dim]
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, embedding_dim, input_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, input_dim)
        )

    def forward(self, z):
        z = self.decoder(z)  # Shape: [Batch * 7, Output Dim]
        return z

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.encoding = self.encoding.unsqueeze(0)  # Shape: [1, max_len, embedding_dim]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, Seq_len, Embedding_dim]
        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
        
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        #self.norm = LayerNormalization()
        self.norm = nn.LayerNorm(normalized_shape=128)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        #self.norm = LayerNormalization()
        self.norm = nn.LayerNorm(normalized_shape=128)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, encoder_output):
        # Apply cross-attention directly using the encoder output
        x = self.residual_connections[0](encoder_output, lambda x: self.cross_attention_block(encoder_output, encoder_output, encoder_output))
        # Apply the feedforward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(normalized_shape=128)

    def forward(self, encoder_output):
        # Pass the encoder output through all layers
        #breakpoint()
        for layer in self.layers:
            x = layer(encoder_output)
        x = self.norm(x)
        return x
