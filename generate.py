import torch
import tiktoken
from model import GPT, GPTConfig

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load the tiktoken tokenizer (gpt2)
enc = tiktoken.get_encoding("gpt2")
vocab_size = 50304 # Matching our 124M model spec

print(f"Initializing model on {device}...")
config = GPTConfig()
model = GPT(config)

print("Loading trained weights...")
try:
    # Try loading the checkpoint if it exists
    checkpoint = torch.load('model_100M_ckpt.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Resumed from iteration {checkpoint['iter']}")
except FileNotFoundError:
    print("No checkpoint found. The model is completely untrained (random weights)!")

model.to(device)
model.eval()

# Generate
prompt = "In the beginning"
print(f"Prompt: {prompt}\n")

# Encode prompt
context = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)

# Generate
generated = model.generate(context, max_new_tokens=200, temperature=0.8)

# Decode
response_tokens = generated[0].tolist()
response_text = enc.decode(response_tokens)

print("Generated Output:\n")
print(response_text)
