import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare_dataset():
    # Use OpenAI's gpt-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # We will download a high-quality slice of Wikipedia (approx 100M-150M tokens)
    # This is roughly 500MB of pure text.
    print("Downloading WikiText-103 dataset (this may take a few minutes)...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    val_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    
    def process_split(split_dataset, filename):
        print(f"Tokenizing {filename}...")
        # To avoid keeping everything in RAM, we write to the bin file in chunks
        arr_len = 0
        
        # Open file in write-binary mode
        with open(filename, 'wb') as f:
            for example in tqdm(split_dataset):
                text = example['text']
                if not text.strip():
                    continue
                # encode text
                tokens = enc.encode_ordinary(text)
                # optionally add an End Of Text token
                tokens.append(enc.eot_token)
                
                # convert to uint16 (gpt2 vocab is 50257, which fits in uint16)
                tokens_np = np.array(tokens, dtype=np.uint16)
                f.write(tokens_np.tobytes())
                arr_len += len(tokens)
                
        print(f"Finished {filename}: {arr_len:,} tokens.")

    process_split(dataset, 'train.bin')
    process_split(val_dataset, 'val.bin')
    
if __name__ == '__main__':
    prepare_dataset()
