import os
import time
import pickle
import numpy as np
import torch
from model import GPT, GPTConfig
from tokenizer import BasicTokenizer

# Hyperparameters for Fine-Tuning
batch_size = 16 # smaller batch size for fine-tuning
block_size = 256
max_iters = 500 # We don't need many iterations for SFT on a tiny dataset
eval_interval = 100
learning_rate = 3e-5 # Lower learning rate so we don't destroy pre-trained weights
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 10

torch.manual_seed(1337)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.vocab)

# Load SFT data
train_data = np.memmap('train_instruct.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val_instruct.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'mps':
        x, y = x.to('mps'), y.to('mps')
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model and load pre-trained weights
config = GPTConfig(vocab_size=vocab_size)
model = GPT(config)
print("Loading pre-trained base model weights...")
model.load_state_dict(torch.load('model.pt', map_location=device))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting Supervised Fine-Tuning (SFT)...")
t0 = time.time()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), 'chat_model.pt')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

t1 = time.time()
print(f"Fine-Tuning completed in {t1 - t0:.2f}s")
print("Saved instruction-tuned model as chat_model.pt")
