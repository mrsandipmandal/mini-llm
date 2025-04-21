import torch.nn as nn

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        x, hidden = self.rnn(x, hidden)
        return self.fc(x), hidden

