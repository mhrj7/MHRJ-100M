# The 100M Parameter Project 

The codebase has been completely overhauled to support training a massive 124 Million parameter model (the exact size of the original GPT-2 Small) for a 10-day run on your Mac Mini M4!

## What Was Accomplished

1. **Massive Architecture Upgrade**: Upgraded [`model.py`](file:///Volumes/MHRJ/scratch_llm/model.py) to feature 12 Transformer Layers, 12 Attention Heads, and a 768-dimensional embedding space.
2. **Robust 10-Day Training Script**: Completely rewrote [`train.py`](file:///Volumes/MHRJ/scratch_llm/train.py) to include:
   - **Gradient Accumulation**: To allow the M4 to handle massive mathematical batches without overflowing its Unified Memory.
   - **Tqdm Progress Bar**: A live progress bar so you can watch the loss go down in real-time in your terminal!
   - **Robust Checkpointing**: It saves a `model_100M_ckpt.pt` file every 2,000 steps. If your Mac reboots or you cancel the script, it will perfectly resume where you left off.
3. **The Data Pipeline**: Wrote [`download_dataset.py`](file:///Volumes/MHRJ/scratch_llm/download_dataset.py) to download a chunk of Wikipedia (`Wikitext-103`), replacing our old 5MB Bible dataset with a massive 500MB dataset. It also now uses OpenAI's lightning-fast `tiktoken` library (written in Rust/C++) to tokenize the 500MB of text in a few minutes, avoiding the weeks-long delay of our from-scratch Python script!

## How to Start Your 10-Day Run!

To watch the training happen live, **you must run these commands in your own terminal window**.

**Step 1: Download & Tokenize the Massive Dataset**
This will download 500MB of Wikipedia and tokenize it. It will take a few minutes.
```bash
cd /Volumes/MHRJ/scratch_llm
source venv/bin/activate
python3 download_dataset.py
```

**Step 2: Start the 100M Training Loop**
```bash
python3 train.py
```
You will immediately see a progress bar tracking the iterations, showing your loss dropping! You can stop the script (`Ctrl+C`) at any time, and when you run `python3 train.py` again, it will automatically resume from the last saved checkpoint!
