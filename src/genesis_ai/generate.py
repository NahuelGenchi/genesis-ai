from __future__ import annotations

import argparse

import torch

from .checkpoint import load_model, tokenizer_from_payload


def generate_text(
    checkpoint: str,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 1337,
    device: str = "cpu",
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        tokens,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    return tokenizer.decode(output, errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a Genesis AI checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k = None if args.top_k <= 0 else args.top_k
    print(
        generate_text(
            args.checkpoint,
            args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
