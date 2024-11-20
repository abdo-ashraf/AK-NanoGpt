import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention_Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_features):
        super().__init__()
        self.head_features = head_features

        self.WQ = nn.Linear(dim_embd, head_features, bias=False)
        self.WK = nn.Linear(dim_embd, head_features, bias=False)
        self.WV = nn.Linear(dim_embd, head_features, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B,T,C = x.shape
        q = self.WQ(x)
        k = self.WK(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        ## this mask to disconnect the token from following tokens in the seqeunce
        ## we will use this mask during training only
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        attention_weights = F.softmax(wei, dim=-1) ## this is attention weights
        attention_weights = self.dropout(attention_weights)

        v = self.WV(x)
        out = attention_weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_features):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention_Head(head_features) for _ in range(num_heads)])
        ## Projection to make output compatible with residual adding
        self.proj = nn.Linear(head_features*num_heads, dim_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features * 4),
            nn.ReLU(),
            ## Projection to make output compatible with residual adding
            nn.Linear(in_features * 4, in_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)
    
## LayerNorm: same as BatchNorm ,but it normalize the rows not the columns
class Block(nn.Module):
    def __init__(self, input_features, num_heads):
        super().__init__()
        head_size = input_features // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(input_features)
        self.ln1 = nn.LayerNorm(input_features)
        self.ln2 = nn.LayerNorm(input_features)

    def forward(self, x):
        ## (x +) is for residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Attentioned_LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = block_size
        self.embd_layers = nn.Embedding(vocab_size, dim_embd)
        self.position_encoding_layer = nn.Embedding(block_size, dim_embd)
        self.blocks = nn.Sequential(*[Block(dim_embd, num_heads=head_size) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(dim_embd)
        self.lm_head = nn.Linear(dim_embd, vocab_size)
        self.apply(self._init_weights)
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        embdings = self.embd_layers(idx)
        pos_encodes = self.position_encoding_layer(torch.arange(T, device=device))
        x = embdings + pos_encodes ## (B,T,dim_embd) + (T,dim_embd) = (B,T,dim_embd) broadcast happened
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, seq_len, embds = logits.shape
            logits = logits.view(batch*seq_len, embds)
            targets = targets.view(batch*seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            # print(logits)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            # print(probs)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

## In this code we still preserve history tokens, however we don't use them.
