from model.transformer import MiniLLM
from torch.utils.data import DataLoader, Dataset
import torch, torch.nn as nn, torch.optim as optim, pickle

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, i):
        x = self.data[i : i+self.seq_len]
        y = self.data[i+1 : i+1+self.seq_len]
        return x, y

def train():
    # 1) Load & preprocess
    with open("data/tiny.txt", encoding="utf-8", errors="ignore") as f:
        raw = f.read(100_000)
    char2idx = {c:i for i,c in enumerate(sorted(set(raw)))}
    idx2char = {i:c for c,i in char2idx.items()}
    data = torch.tensor([char2idx[c] for c in raw])
    seq_len = 32

    # 2) DataLoader
    dataset = CharDataset(data, seq_len)
    loader  = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # 3) Model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = MiniLLM(vocab_size=len(char2idx), embed_dim=64).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt     = optim.Adam(model.parameters(), lr=3e-3)

    # 4) Training loop
    for epoch in range(20):
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            out, _ = model(x_batch)
            loss = loss_fn(out.view(-1, out.size(-1)), y_batch.view(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/20 – avg loss {total/len(loader):.4f}")

    # 5) Save model + vocab
    torch.save({
        'model_state': model.state_dict(),
        'char2idx':    char2idx,
        'idx2char':    idx2char
    }, "mini_model.pth")
    print("Training complete.")

if __name__ == "__main__":
    train()