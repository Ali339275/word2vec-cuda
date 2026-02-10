import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        self.in_embed.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.out_embed.weight.data.zero_()

    def forward(self, center_words, pos_words, neg_words):
        # center_words: (batch,)
        # pos_words: (batch,)
        # neg_words: (batch, K)

        v = self.in_embed(center_words)          # (batch, emb_dim)
        u_pos = self.out_embed(pos_words)        # (batch, emb_dim)
        u_neg = self.out_embed(neg_words)        # (batch, K, emb_dim)

        # positive score
        pos_score = torch.sum(v * u_pos, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # negative score
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss + neg_loss)
        return loss.mean()