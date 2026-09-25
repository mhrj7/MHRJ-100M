# MHRJ-100M: A 124M Parameter Decoder-Only Transformer

MHRJ-100M is a custom, from-scratch Large Language Model (LLM) developed entirely in Python. The project demonstrates a complete end-to-end pipeline, starting from custom data ingestion and tokenization to building a GPT-2 style neural network architecture, and executing a massive 10-day training run on local hardware. 

This repository showcases the ability to architect, optimize, and train an industrial-scale language model utilizing both PyTorch and the TensorFlow ecosystem, successfully bridging the gap between theoretical deep learning and practical hardware optimization.

## 🧠 Architecture Overview

At its core, MHRJ-100M is a **Decoder-Only Transformer** modeled after the original OpenAI GPT-1 / GPT-2 Small specifications.

* **Parameters:** 124.39 Million
* **Transformer Blocks (Layers):** 12
* **Attention Heads:** 12
* **Embedding Dimension:** 768
* **Context Window (Block Size):** 1024 Tokens
* **Vocabulary Size:** 50,304 (Optimized padded vocabulary)
* **Core Components:** Causal Self-Attention, Multi-Layer Perceptrons (GELU activations), and robust Layer Normalization.

## 🛠️ Framework Split: PyTorch & TensorFlow

To maximize efficiency and demonstrate framework fluency, this project utilizes a dual-framework approach:

1. **PyTorch (Core Architecture & Training):** 
   The underlying Transformer architecture (`model.py`), custom training loop, loss calculation, and backpropagation (`train.py`) are built purely in PyTorch. PyTorch's dynamic computational graph was strictly required for the low-level optimizations implemented (like MPS synchronization).
2. **TensorFlow (Data Pipeline & Preprocessing):** 
   While PyTorch handles the GPU math, **TensorFlow (`tf.data`)** was utilized for the heavy-duty data preprocessing pipeline. Parsing, shuffling, and streaming a massive corpus of text requires robust multi-threading and pre-fetching mechanisms. The TensorFlow ecosystem efficiently pipelines the gigabytes of raw text data from disk into tokenized tensors before passing them off to the PyTorch training loop, preventing the CPU from bottlenecking the GPU.

## 📚 Training Data & Tokenization

* **Dataset:** WikiText-103 (A massive subset of high-quality Wikipedia articles).
* **Corpus Size:** ~500 MB of pure text.
* **Token Count:** ~119,000,000 Tokens.
* **Tokenization:** Initially implemented as a from-scratch Byte-Pair Encoding (BPE) algorithm (`tokenizer.py`) for educational purposes. To scale to 119M tokens, the data pipeline was upgraded to utilize OpenAI's highly-optimized C++/Rust `tiktoken` (GPT-2 encoding) to process the entire dataset in under 30 seconds.

## 💻 Compute Infrastructure & Hardware Optimization

The model is currently being trained entirely on local hardware: an **Apple Mac Mini M4**.

Training a 124M parameter model on a unified memory architecture (without a massive datacenter cluster) required severe memory optimizations:
1. **MPS Backend:** Leverages Apple's Metal Performance Shaders (`torch.backends.mps`) for hardware-accelerated GPU compute.
2. **Gradient Accumulation:** To prevent RAM overflows and SSD-swapping, the batch size is kept micro-small, while gradients are accumulated over 8 to 32 micro-steps to simulate a massive global batch size.
3. **Forced Synchronization:** Implemented forced MPS queue synchronization (`loss.item()` trapping) to prevent asynchronous operation pileups from crashing the system.
4. **Robust Checkpointing:** Features automatic state-saving every 2,000 iterations to ensure the 10-day training run can survive power outages or manual pauses.

## 📈 Current Status

**Status: In Progress (Training Phase)**
The model is currently executing a 10-day continuous pre-training loop. 
* **Initial Loss:** 4.60
* **Current Trajectory:** Steadily converging down towards the 2.x range as it learns the statistical structures of the English language. 

Once pre-training concludes, the model will proceed to the **Supervised Fine-Tuning (SFT)** phase (`finetune.py`) using a conversational Q&A dataset to transform it from a base completion model into an interactive chatbot (`chat.py`).
