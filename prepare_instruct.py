import json
import pickle
import torch
import numpy as np
from tokenizer import BasicTokenizer

def prepare_instruct_data():
    with open("instruct.json", "r") as f:
        dataset = json.load(f)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Format the dataset into a continuous text stream
    # A real instruction dataset would use special EOS tokens and masking, 
    # but for this basic demonstration, we just format it as a continuous script.
    formatted_text = ""
    for pair in dataset:
        formatted_text += f"User: {pair['user']}\nAI: {pair['ai']}\n\n"

    print(f"Formatted text length: {len(formatted_text)} characters.")

    # Encode
    data_ids = tokenizer.encode(formatted_text)
    print(f"Instruction dataset has {len(data_ids):,} tokens")

    # Split
    data = torch.tensor(data_ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Save
    train_data.numpy().astype(np.uint16).tofile('train_instruct.bin')
    val_data.numpy().astype(np.uint16).tofile('val_instruct.bin')
    print("Saved train_instruct.bin and val_instruct.bin")

if __name__ == '__main__':
    prepare_instruct_data()
