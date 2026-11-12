"""Fine-tuning script for the retrieval-augmented summariser."""

import argparse
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.planner import SentencePlanner


def train(data_path: str, epochs: int = 8):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    backbone = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = SentencePlanner(backbone)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        losses = []
        for batch in load_batches(data_path, tokenizer):
            tokens, _ = model(**batch)
            loss = torch.nn.functional.cross_entropy(tokens, batch["labels"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        print(epoch, sum(losses) / len(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(args.data, args.epochs)
