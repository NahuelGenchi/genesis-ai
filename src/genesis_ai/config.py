from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 256
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.0
    position_encoding: str = "learned"

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("n_layers and d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.position_encoding not in {"learned", "rotary"}:
            raise ValueError("position_encoding must be learned or rotary")
        if self.position_encoding == "rotary" and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("rotary position encoding requires an even head dimension")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        config = cls(**values)
        config.validate()
        return config
