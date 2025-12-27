import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for query vectors.
    """
    def __init__(self, dim, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        # x: [Batch, Length, Dim]
        # Broadcast PE across batch
        return x + self.pe[:, :x.size(1), :]

class MHSA(nn.Module):
    """
    Multi-Head Self-Attention wrapper to implement Eq. 4, 5, 7.
    """
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super(MHSA, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Pre-Norm architecture for stability
        q_norm = self.norm(query)
        k_norm = self.norm(key)
        v_norm = self.norm(value)
        
        attn_output, _ = self.self_attn(q_norm, k_norm, v_norm)
        # Residual connection
        return query + self.dropout(attn_output)