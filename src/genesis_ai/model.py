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


class DenseFFN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_experts = config.moe_experts
        self.top_k = config.moe_top_k
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_ff, bias=False),
                    nn.GELU(),
                    nn.Linear(config.d_ff, config.d_model, bias=False),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.register_buffer("routing_counts", torch.zeros(self.n_experts, dtype=torch.long), persistent=False)
        self.last_aux_loss: torch.Tensor | None = None

    def reset_routing_stats(self) -> None:
        self.routing_counts.zero_()

    def routing_metrics(self) -> dict[str, object]:
        counts = self.routing_counts.detach().cpu()
        total = int(counts.sum())
        if total == 0:
            fractions = [0.0 for _ in range(self.n_experts)]
        else:
            fractions = [float(value) / total for value in counts.tolist()]
        return {
            "experts": self.n_experts,
            "top_k": self.top_k,
            "assignments": total,
            "counts": counts.tolist(),
            "fractions": fractions,
            "utilization": sum(value > 0 for value in counts.tolist()) / self.n_experts,
            "max_fraction": max(fractions) if fractions else 0.0,
            "min_fraction": min(fractions) if fractions else 0.0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        if self.training:
            counts = torch.bincount(top_indices.detach().reshape(-1), minlength=self.n_experts)
            self.routing_counts.add_(counts.to(self.routing_counts.device))

        load = F.one_hot(top_indices, num_classes=self.n_experts).to(router_probs.dtype).mean(dim=(0, 1))
        importance = router_probs.mean(dim=0)
        self.last_aux_loss = self.n_experts * torch.sum(load.detach() * importance)

        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_positions, slots = torch.where(top_indices == expert_index)
            if token_positions.numel() == 0:
                continue
            expert_output = expert(flat[token_positions])
            weighted = expert_output * top_weights[token_positions, slots].unsqueeze(-1)
            output.index_add_(0, token_positions, weighted)
        return output.view(original_shape)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp: nn.Module = SparseMoE(config) if config.ffn_type == "moe" else DenseFFN(config)
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

    def reset_routing_stats(self) -> None:
        for block in self.blocks:
            if isinstance(block.mlp, SparseMoE):
                block.mlp.reset_routing_stats()

    def routing_metrics(self) -> list[dict[str, object]]:
        return [block.mlp.routing_metrics() for block in self.blocks if isinstance(block.mlp, SparseMoE)]

    def _router_aux_loss(self) -> torch.Tensor | None:
        losses = [
            block.mlp.last_aux_loss
            for block in self.blocks
            if isinstance(block.mlp, SparseMoE) and block.mlp.last_aux_loss is not None
        ]
        if not losses:
            return None
        return torch.stack(losses).mean()

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
            if self.training and self.config.ffn_type == "moe" and self.config.moe_aux_loss_weight > 0:
                aux_loss = self._router_aux_loss()
                if aux_loss is not None:
                    loss = loss + self.config.moe_aux_loss_weight * aux_loss
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

    def estimated_active_parameter_count(self) -> int:
        total = self.parameter_count()
        if self.config.ffn_type != "moe":
            return total
        expert_parameters = 2 * self.config.d_model * self.config.d_ff
        inactive_per_layer = (self.config.moe_experts - self.config.moe_top_k) * expert_parameters
        return total - self.config.n_layers * inactive_per_layer

    def estimated_active_flops_per_token(self) -> int:
        return 6 * self.estimated_active_parameter_count()
