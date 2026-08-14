import argparse

import torch

from .checkpoint import load_model


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
    torch.manual_seed(args.seed)
    model, _ = load_model(args.checkpoint, args.device)
    model.eval()
    prompt = args.prompt.encode("utf-8")
    tokens = torch.tensor([list(prompt)], dtype=torch.long, device=args.device)
    output = model.generate(
        tokens,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print(bytes(output).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
