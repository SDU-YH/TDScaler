import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    pass

    def __init__(
        self,
        num_heads: int,
        embed_size: int,
        history_len: int,
        dropout: float,
        attn_gate,
        mlp_gate,
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.attn_gate = attn_gate
        self.mlp_gate = mlp_gate
                                                              
        self.alpha = None

        self.attn_mask = nn.Parameter(
            torch.triu(torch.ones(history_len, history_len), diagonal=1),
            requires_grad=False,
        )
        self.attn_mask[self.attn_mask.bool()] = -float("inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, self.alpha = self.attention(
            x,
            x,
            x,
            attn_mask=self.attn_mask[: x.size(1), : x.size(1)],
            average_attn_weights=True,                                              
        )
                                        
        x = self.attn_gate(x, F.relu(attention))
        x = self.layernorm1(x)
        ffn = self.ffn(x)
                                        
        x = self.mlp_gate(x, F.relu(ffn))
        x = self.layernorm2(x)
        return x

class TransformerIdentityLayer(TransformerLayer):
    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.layernorm1(x)
        attention, self.alpha = self.attention(
            x_norm1,
            x_norm1,
            x_norm1,
            attn_mask=self.attn_mask[: x_norm1.size(1), : x_norm1.size(1)],
            average_attn_weights=True,                                              
        )
                         
        x = self.attn_gate(x, F.relu(attention))
        x_norm2 = self.layernorm2(x)
        ffn = self.ffn(x_norm2)
                         
        x = self.mlp_gate(x, F.relu(ffn))
        return x
