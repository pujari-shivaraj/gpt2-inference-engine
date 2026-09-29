from torch.profiler import profile, ProfilerActivity
from engine.engine import InferenceEngine

engine = InferenceEngine()

# warmup -- same reasoning as before, first call includes one-time costs
engine.generate("The future of AI", max_new_tokens=5)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    engine.generate("The future of AI", max_new_tokens=20)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))