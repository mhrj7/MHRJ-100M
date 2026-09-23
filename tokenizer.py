import collections

class BasicTokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def train(self, text, vocab_size):
        # Convert text to raw bytes (integers 0-255)
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            # Count the frequency of each adjacent pair
            stats = collections.Counter(zip(tokens, tokens[1:]))
            if not stats:
                break
                
            # Find the most common pair
            best_pair = max(stats, key=stats.get)
            
            # The new token ID
            new_id = 256 + i
            
            # Print progress every 100 merges
            if (i+1) % 100 == 0 or i == num_merges - 1:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[best_pair[0]] + self.vocab[best_pair[1]]})")
                
            # Merge the best pair in the tokens list
            new_tokens = []
            i_tok = 0
            while i_tok < len(tokens):
                if i_tok < len(tokens) - 1 and tokens[i_tok] == best_pair[0] and tokens[i_tok+1] == best_pair[1]:
                    new_tokens.append(new_id)
                    i_tok += 2
                else:
                    new_tokens.append(tokens[i_tok])
                    i_tok += 1
                    
            tokens = new_tokens
            
            # Save the merge rule and new vocabulary token
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
    def encode(self, text):
        # Given a string, return list of token IDs
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            stats = collections.Counter(zip(tokens, tokens[1:]))
            # Find the pair that is eligible to be merged (has the lowest merged ID)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # nothing else can be merged
                
            # Merge the best pair
            new_id = self.merges[pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens
        
    def decode(self, ids):
        # Given a list of token IDs, return a string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")
