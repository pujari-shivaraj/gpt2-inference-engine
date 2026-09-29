import torch
import transformers
from engine.engine import InferenceEngine

print("transformers version:", transformers.__version__)

engine = InferenceEngine()
input_ids = engine.tokenizer.encode("The future of AI")
print("input_ids shape:", input_ids.shape)

logits, cache = engine.model.forward(input_ids)
print("cache type:", type(cache))

print("number of layers:", len(cache))
print("sequence length stored:", cache.get_seq_length())

keys, values = [], []

if hasattr(cache, "layers"):
    print("found: cache.layers")
    for layer in cache.layers:
        keys.append(layer.keys)
        values.append(layer.values)
elif hasattr(cache, "key_cache"):
    print("found: cache.key_cache")
    keys = list(cache.key_cache)
    values = list(cache.value_cache)
else:
    print("found: indexing cache[i]")
    for i in range(len(cache)):
        k, v = cache[i]
        keys.append(k)
        values.append(v)

print("layer 0 K shape:", keys[0].shape)
print("layer 0 V shape:", values[0].shape)

total_bytes = sum(t.numel() * t.element_size() for t in keys + values)
print("total cache bytes:", total_bytes)
print("total cache MB:", total_bytes / (1024 ** 2))