import torch
from model.transformer import MiniLLM

# Load checkpoint
checkpoint = torch.load("mini_model.pth")
char2idx = checkpoint["char2idx"]
idx2char = checkpoint["idx2char"]
vocab_size = len(char2idx)

# Rebuild model
model = MiniLLM(vocab_size, embed_dim=32)
model.load_state_dict(checkpoint["model"])
model.eval()

# Generate
input_text = "th"  # start with something common in your dataset
input_ids = torch.tensor([[char2idx[c] for c in input_text]])
generated = input_text
hidden = None
for _ in range(100):
    out, hidden = model(input_ids, hidden)
    next_id = torch.argmax(out[:, -1, :], dim=-1).item()
    next_char = idx2char[next_id]
    generated += next_char
    input_ids = torch.tensor([[next_id]])

print("Generated text:")
print(generated)
