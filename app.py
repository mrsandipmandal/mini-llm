from flask import Flask, render_template, request
import torch
from model.transformer import MiniLLM

app = Flask(__name__)

# ——————— Load the checkpoint (model + vocab) ———————
checkpoint = torch.load("mini_model.pth", map_location="cpu")
char2idx  = checkpoint["char2idx"]
idx2char  = checkpoint["idx2char"]
vocab_size = len(char2idx)

# ——————— Build & load the model ———————
model = MiniLLM(vocab_size, embed_dim=64)             # ← match embed_dim you trained with
model.load_state_dict(checkpoint["model_state"])
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    generated = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        if all(c in char2idx for c in prompt):
            # encoding
            input_ids = torch.tensor([[char2idx[c] for c in prompt]])
            hidden    = None
            generated = prompt
            # generation loop
            for _ in range(100):
                out, hidden = model(input_ids, hidden)
                next_id     = torch.argmax(out[:, -1, :], dim=-1).item()
                next_char   = idx2char[next_id]
                generated  += next_char
                input_ids   = torch.tensor([[next_id]])
        else:
            generated = "Invalid characters in input."
    return render_template("index.html", generated=generated)

if __name__ == "__main__":
    app.run(debug=True)
