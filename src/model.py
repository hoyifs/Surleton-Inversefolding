import os 
import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = 3000  
        pe = torch.zeros(self.max_len, embed_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len]  # [1, seq_len, embed_dim]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=attn_mask)
        x = self.dropout(x)
        x = residual + x  # Residual connection

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x  # Residual connection

        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers, dropout):
        super().__init__()
        

        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4), 
            nn.GELU(),
            nn.Linear(embed_dim // 4, 20)
        )


    def forward(self, x, attn_mask=None):

        positions = self.embed_positions(x)

        x = x + positions
        
        for layer in self.layers:
            x = layer(x, attn_mask)

        x = x.view(-1, x.size(-1))
        attn_mask = attn_mask.view(-1)
        x = x[attn_mask]

        x = self.output_projection(x)
        
        return x

class TransformerModelold(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers, dropout):
        super().__init__()

        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(embed_dim, 20)  

    def forward(self, x, attn_mask=None):

        positions = self.embed_positions(x)

        x = x + positions
        
        for layer in self.layers:
            x = layer(x, attn_mask)

        x = x.view(-1, x.size(-1))
        attn_mask = attn_mask.view(-1)
        x = x[attn_mask]

        x = self.output_projection(x)
        
        return x


class Surfeleton(nn.Module):
    def __init__(self, atomsurf_model, transformer_model, surface_ablation=False, graph_ablation=False):
        super().__init__()
        self.atsf = atomsurf_model
        self.tsfm = transformer_model
        self.sa = surface_ablation
        self.ga = graph_ablation
    def split_batch(self, graph, device):
        x_splits = []
        attn_masks = []
        glen = graph.node_len
        max_len = max(glen)
        start = 0
        for length in graph.node_len:
            end = start + length
            x = graph.x[start:end]

            padding_size = max_len - length
            padded_x = F.pad(x, (0, 0, 0, padding_size))  
            x_splits.append(padded_x)

            mask = torch.ones(max_len + 1)
            mask[length:] = 0
            attn_masks.append(mask.bool())

            start = end


        x_batch = torch.stack(x_splits)
        padding_column = torch.zeros((x_batch.size(0), 1, x_batch.size(2))).to(device)
        x_batch = torch.cat([x_batch, padding_column], dim=1)

        attn_mask = torch.stack(attn_masks)

        return x_batch, attn_mask.to(device), glen


    def forward(self, pro_batch, device):

        if self.sa:
            pro_batch.surface.x = torch.ones_like(pro_batch.surface.x)

        if self.ga:
            pro_batch.graph.x = torch.ones_like(pro_batch.graph.x)

        _, graph = self.atsf(graph=pro_batch.graph, surface=pro_batch.surface)

        s_batch, mask, glen = self.split_batch(graph, device)

        output_pre = self.tsfm(s_batch, attn_mask=mask)

        return output_pre, glen

