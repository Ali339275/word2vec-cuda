import time
import torch
from torch.optim import Adam
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataloader, epochs, lr=0.001, device="cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device

        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self):
        self.model.train()

        # -----------------------------
        # START TIMING
        # -----------------------------
        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        for epoch in range(self.epochs):
            total_loss = 0.0
            loop = tqdm(self.dataloader, desc=f"Epoch [{epoch+1}/{self.epochs}]")

            for center, pos, neg in loop:
                center = center.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(center, pos, neg)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=total_loss / (loop.n + 1))

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        # -----------------------------
        # END TIMING
        # -----------------------------
        if self.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds")