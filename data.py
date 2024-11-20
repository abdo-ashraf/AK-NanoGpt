import torch

# !wget -q https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('./input.txt') as f:
    text = f.read()
print(f"{len(text) = } characters", end='\n-------------Data-------------\n')
print(text[:100])

chars = sorted(list(set(text)))
stoi = {ch:idx for idx, ch in enumerate(chars)}
itos = {idx:ch for idx, ch in enumerate(chars)}
vocab_size = len(chars)
print(vocab_size)

encode = lambda text: [stoi[c] for c in text]
decode = lambda tokens: ''.join([itos[c] for c in tokens])
raw = 'i love this game.'
tokens = encode(raw)
print(tokens)
print(decode(tokens))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.type())
print(data[:100])

def get_batch(split_type):
    data = train_data if split_type=='train' else val_data
    ix = torch.randint(low=0, high=len(data)-block_size, size=(batch_size,), generator=g)
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y
xb, yb = get_batch('train')
print(xb.shape, yb.shape)

torch.save(stoi, './stoi.pt')