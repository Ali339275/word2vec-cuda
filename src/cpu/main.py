import torch
from torch.utils.data import DataLoader

from model import SkipGram
from data.dataset import SkipGramDataset   # your dataset code
from train import Trainer


def main():
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    embedding_dim = 288
    window_size = 1
    num_negatives = 60
    epochs = 15
    lr = 0.01
    batch_size = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load data
    # -----------------------------
    dataset = SkipGramDataset(
        text_path="data/text8_500k",
        window_size=window_size,
        num_negatives=num_negatives
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = SkipGram(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
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