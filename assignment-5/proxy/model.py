"""A minimal GPT (nanoGPT-style), sized to train on a CPU. ~5-12M params."""
import math, torch, torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 16000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 288
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd); self.ln2 = nn.LayerNorm(c.n_embd)
        self.attn = nn.MultiheadAttention(c.n_embd, c.n_head, dropout=c.dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout))
        self.register_buffer("mask", torch.triu(torch.ones(c.block_size, c.block_size) * float("-inf"), 1))

    def forward(self, x):
        T = x.size(1)
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.lnf = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.tok.weight = self.head.weight            # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks: x = b(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
