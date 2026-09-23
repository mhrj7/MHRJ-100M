import pickle
import torch
from model import GPT, GPTConfig
from tokenizer import BasicTokenizer
import sys

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.vocab)

print(f"Initializing Instruction-Tuned model on {device}...")
config = GPTConfig(vocab_size=vocab_size)
model = GPT(config)

print("Loading trained chat_model.pt weights...")
# Try loading the chat model. If it doesn't exist, fall back to base model for safety
try:
    model.load_state_dict(torch.load('chat_model.pt', map_location=device))
except FileNotFoundError:
    print("chat_model.pt not found, loading base model.pt instead.")
    model.load_state_dict(torch.load('model.pt', map_location=device))
    
model.to(device)
model.eval()

print("="*50)
print("Instruction-Tuned AI Chatbot Ready!")
print("Type 'quit' or 'exit' to leave.")
print("="*50)

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        print("\nAI:", end=" ")
        sys.stdout.flush()
        
        # Format the prompt using the conversational template we trained on
        prompt = f"User: {user_input}\nAI:"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        generated = model.generate(context, max_new_tokens=150, temperature=0.7)
        
        # Decode only the newly generated tokens
        response_tokens = generated[0][len(context[0]):].tolist()
        response_text = tokenizer.decode(response_tokens)
        
        # Stop printing if the model accidentally generates another "User:" tag
        if "User:" in response_text:
            response_text = response_text.split("User:")[0]
            
        print(response_text.strip())
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
