

"""
from engine.engine import InferenceEngine
from benchmarks.benchmark import compare

engine = InferenceEngine()

compare(engine, "The future of AI")


from engine.engine import InferenceEngine
from benchmarks.benchmark import sweep

engine = InferenceEngine()
sweep(engine)
"""
from engine.engine import InferenceEngine
from batching.build_batch import build_batch

engine = InferenceEngine()

prompts = ["Explain transformers", "What is Python?", "Write a short poem."]
input_ids, attention_mask = build_batch(engine, prompts)

print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)
print(input_ids)
print(attention_mask)
