import json
import math
import torch
import numpy as np
from load_magic_checkpoint import load_magic_checkpoint
from modeling_magic import load_magic_config

DATA_PATH = "Magic-Base-Data.json"
CKPT_DIR = "checkpoints/magic_base"
DEVICE = "cpu"

cfg = load_magic_config("magic_config.yaml")

model, input_proj, lm_head, scaler, le, _ = load_magic_checkpoint(
    cfg, CKPT_DIR, DEVICE
)

with open(DATA_PATH) as f:
    data = json.load(f)

X = np.array([[len(d["input"]), sum(c.isupper() for c in d["input"])] for d in data])
y = np.arange(len(data))

X = torch.tensor(scaler.transform(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

with torch.no_grad():
    logits = lm_head(input_proj(X))
    loss = torch.nn.CrossEntropyLoss()(logits, y).item()
    ppl = math.exp(loss)

print("📊 MAGIC BASE EVAL")
print("Loss:", loss)
print("Perplexity:", ppl)
