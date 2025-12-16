import torch
import torch.nn as nn
from torch.nn import functional as F

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Running on {torch.cuda.get_device_name(0)}")
else: # ATTENTION: run the file on CPU and you will cook!
    device = torch.device("cpu")
    print("Running on CPU")
    raise SystemExit('You cooked!')
    
# Hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
lr = 3e-4
max_new_tokens = 500
n_embd = 384
n_head = 6 # communication channel
n_layer = 6
dropout = 0.2 # prevent overfitting
 
torch.manual_seed(1337)

# Open file
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = open('input.txt', 'r', encoding='utf-8').read()

# Encode - decode
chars = sorted(list(set(text)))
vocab_size = len(chars)
c2i = {c:i for i, c in enumerate(chars)}
i2c = {i:c for i, c in enumerate(chars)}
encode = lambda s: [c2i[c] for c in s] # take string -> list of int
decode = lambda li: ''.join([i2c[i] for i in li]) # take list of int -> string

# Train, val split
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

class SelfAttHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        k = self.key(x)
        B, T, C = k.shape
        q = self.query(x)
        v = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, C)
        return out

class MultiAttHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_embd)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    # a simple Linear layer with non-linearity
    def __init__(self, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_hidden, 4 * n_hidden), 
            nn.ReLU(),
            nn.Linear(4 * n_hidden, n_hidden),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiAttHead(n_head, n_embd//n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual and ln before add
        x = x + self.ffwd(self.ln2(x)) # residual and ln before add
        return x

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiAttHead(n_head, n_embd//n_head) # 4 heads of 8-dim self-att = 32 in total
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb_table(idx) # (B,T,C) (Batch, Time, Channel)
        pos_emb = self.pos_emb_table(torch.arange(end=T, device=device)) # (T, C)
        out = tok_emb + pos_emb
        out = self.blocks(out)
        out = self.ln_f(out)
        logits = self.lm_head(out) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate (self, idx, max_new_tokens):
        # idx is [B, T] array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # [B,T,C]
            logits = logits[:,-1,:] # last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim =-1)
        return idx
    
model = BigramLM()
model = model.to(device)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print (f"step {step}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.int64, device=device)
for s in model.generate(idx, max_new_tokens).tolist():
    print (decode(s))