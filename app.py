from flask import Flask, render_template, request
import torch
from model.transformer import MiniLLM
import pickle

app = Flask(__name__)

# Load vocab
with open("vocab.pkl", "rb") as f:
    char2idx, idx2char = pickle.load(f)

vocab_size = len(char2idx)
model = MiniLLM(vocab_size)
model.load_state_dict(torch.load("mini_model.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    generated = ""
    if request.method == "POST":
        user_input = request.form["prompt"]
        if all(c in char2idx for c in user_input):
            input_ids = torch.tensor([[char2idx[c] for c in user_input]])
            hidden = None
            generated = user_input
            for _ in range(100):
                out, hidden = model(input_ids, hidden)
                next_id = torch.argmax(out[:, -1, :], dim=-1).item()
                next_char = idx2char[next_id]
                generated += next_char
                input_ids = torch.tensor([[next_id]])
        else:
            generated = "Invalid characters in input."
    return render_template("index.html", generated=generated)

if __name__ == "__main__":
    app.run(debug=True)
