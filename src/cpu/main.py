import argparse
import torch
from torch.utils.data import DataLoader

from model import SkipGram
from data.dataset import SkipGramDataset
from train import Trainer


def main():
    # -----------------------------
    # Argument Parser
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Train Skip-Gram with Negative Sampling"
    )

    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Embedding dimension size")

    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")

    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")

    parser.add_argument("--window_size", type=int, default=1,
                        help="Context window size")

    parser.add_argument("--num_negatives", type=int, default=60,
                        help="Number of negative samples")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load data
    # -----------------------------
    dataset = SkipGramDataset(
        text_path="data/text8_500k",
        window_size=args.window_size,
        num_negatives=args.num_negatives
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = SkipGram(
        vocab_size=dataset.vocab_size,
        embedding_dim=args.embedding_dim
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    trainer.train()

    # -----------------------------
    # Save embeddings
    # -----------------------------
    torch.save(
        model.in_embed.weight.data.cpu(),
        "word_embeddings.pt"
    )


if __name__ == "__main__":
    main()