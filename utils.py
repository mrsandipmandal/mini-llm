import torch

def build_vocab(text):
    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def encode(text, char2idx):
    return [char2idx[c] for c in text]

def decode(indices, idx2char):
    return ''.join([idx2char[i] for i in indices])

def get_dataset(text, seq_len, char2idx):
    data = encode(text, char2idx)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+1:i+seq_len+1])
    return torch.tensor(X), torch.tensor(y)
