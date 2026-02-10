import torch
import torch.nn.functional as F
from data.dataset import SkipGramDataset

# -----------------------------
# Load dataset (for vocab only)
# -----------------------------
dataset = SkipGramDataset(
    text_path="data/text8_1M",
    window_size=2,
    num_negatives=50
)

word2id = dataset.word2id
id2word = dataset.id2word

# -----------------------------
# Load trained embeddings
# -----------------------------
emb = torch.load("word_embeddings.pt")  # shape: (vocab_size, emb_dim)

# -----------------------------
# Nearest neighbors
# -----------------------------
def nearest(word, k=5):
    if word not in word2id:
        return f"'{word}' not in vocabulary"

    v = emb[word2id[word]]
    sims = F.cosine_similarity(v.unsqueeze(0), emb)
    top = sims.topk(k + 1).indices.tolist()

    return [id2word[i] for i in top if id2word[i] != word][:k]


print("man:", nearest("man"))
print("town:", nearest("town"))
print("numbers:", nearest("numbers"))