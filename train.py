import os
import time
import math
import pickle
import numpy as np
import torch
from model import GPT, GPTConfig
from tqdm import tqdm

# Hyperparameters for 124M param 10-day run
batch_size = 2 # Reduced to prevent RAM overload
gradient_accumulation_steps = 8 # Reduced to speed up progress bar ticks
block_size = 1024
max_iters = 100000 # Massively increased for a long run
eval_interval = 2000 # Save checkpoint every 2000 steps
learning_rate = 6e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 20

torch.manual_seed(1337)

# Load data
print("Loading massive dataset (this may take a moment)...")
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
print(f"Train dataset size: {len(train_data):,} tokens")

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
def estimate_loss(model):
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

# Initialize model
config = GPTConfig()
model = GPT(config)
model.to(device)

print(f"Model initialized on {device}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Resume from checkpoint if it exists
start_iter = 0
checkpoint_path = 'model_100M_ckpt.pt'
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter']
    print(f"Resumed at iteration {start_iter}")

# Training loop
print("Starting heavy-duty training loop...")
model.train()
t0 = time.time()

# Using tqdm so you can watch the progress in your terminal!
pbar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters, desc="Training")

for iter in pbar:
    
    # Evaluate and save checkpoint
    if iter % eval_interval == 0 and iter > start_iter:
        losses = estimate_loss(model)
        pbar.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Robust Checkpointing
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter': iter,
            'train_loss': losses['train'].item(),
            'val_loss': losses['val'].item(),
        }
        torch.save(checkpoint, checkpoint_path)
        pbar.write(f"Checkpoint saved to {checkpoint_path}")

    accumulated_loss = 0.0
    # Gradient Accumulation Loop
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        # Scale the loss to account for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Force MPS synchronization to prevent memory overflow and SSD swapping
        accumulated_loss += loss.item()

    # Step the optimizer once every gradient_accumulation_steps
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Update progress bar description with loss
    pbar.set_description(f"Loss: {accumulated_loss:.4f}")

t1 = time.time()
print(f"Training completed in {(t1 - t0)/3600:.2f} hours")
