import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.position_encoding = config.position_encoding
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        if self.position_encoding == "rotary":
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
            self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("rotary_inv_freq", torch.empty(0), persistent=False)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time = q.size(-2)
        positions = torch.arange(time, device=q.device, dtype=self.rotary_inv_freq.dtype)
        angles = torch.outer(positions, self.rotary_inv_freq)
        cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            even = x[..., 0::2]
            odd = x[..., 1::2]
            rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
            return rotated.flatten(-2)

        return rotate(q), rotate(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        if self.position_encoding == "rotary":
            q, k = self._apply_rotary(q, k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class GenesisLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.context_length, config.d_model)
            if config.position_encoding == "learned"
            else None
        )
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        _, time = tokens.shape
        if time > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        x = self.token_embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(time, device=tokens.device)
            x = x + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, k).values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def estimated_active_flops_per_token(self) -> int:
        # Coarse dense-transformer estimate for relative experiment tracking.
        return 6 * self.parameter_count()
