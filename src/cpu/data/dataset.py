import torch
from torch.utils.data import Dataset
from collections import Counter
import random
import math


class SkipGramDataset(Dataset):
    """
    Skip-gram dataset with:
    - subsampling of frequent words
    - on-the-fly (center, context) generation
    - negative sampling
    """

    def __init__(
        self,
        text_path,
        window_size,
        num_negatives,
        min_count=5,
        subsample_t=1e-5,
        neg_sampling_power=0.75
    ):
        self.window_size = window_size
        self.num_negatives = num_negatives

        # -----------------------------
        # Load text
        # -----------------------------
        with open(text_path, "r") as f:
            words = f.read().strip().split()

        # -----------------------------
        # Build vocabulary
        # -----------------------------
        counter = Counter(words)
        self.vocab = [w for w, c in counter.items() if c >= min_count]

        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(self.vocab)

        # Filter words by vocab
        words = [w for w in words if w in self.word2id]

        # -----------------------------
        # Subsampling probabilities
        # -----------------------------
        total_count = sum(counter[w] for w in self.vocab)

        self.keep_prob = {}
        for w in self.vocab:
            f = counter[w] / total_count
            self.keep_prob[w] = min(
                1.0,
                math.sqrt(subsample_t / f) + subsample_t / f
            )

        # Apply subsampling
        self.words = [
            w for w in words
            if random.random() < self.keep_prob[w]
        ]

        # Convert to ids
        self.word_ids = [self.word2id[w] for w in self.words]

        # -----------------------------
        # Negative sampling distribution
        # -----------------------------
        freqs = torch.tensor(
            [counter[self.id2word[i]] for i in range(self.vocab_size)],
            dtype=torch.float
        )
        self.neg_dist = freqs.pow(neg_sampling_power)
        self.neg_dist /= self.neg_dist.sum()

    def __len__(self):
        # Each word can act as a center
        return len(self.word_ids)

    def __getitem__(self, idx):
        center = self.word_ids[idx]

        # Dynamic window (Word2Vec style)
        window = random.randint(1, self.window_size)

        context_ids = []
        for j in range(-window, window + 1):
            if j == 0:
                continue
            pos = idx + j
            if 0 <= pos < len(self.word_ids):
                context_ids.append(self.word_ids[pos])

        # Randomly choose one context word
        pos_word = random.choice(context_ids)

        # Negative sampling
        neg_words = torch.multinomial(
            self.neg_dist,
            self.num_negatives,
            replacement=True
        )

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(pos_word, dtype=torch.long),
            neg_words
        )