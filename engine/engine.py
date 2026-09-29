"""



from model.gpt2_model import GPT2Model
from tokenizer.gpt2_tokenizer import GPT2Tokenizer
import torch
from generation.sampling import greedy_sample


class InferenceEngine:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2Model(model_name)
        self.tokenizer = GPT2Tokenizer(model_name)
    

    def generate_naive(self, prompt, max_new_tokens=50):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = input_ids.to(self.model.device)

        for step in range(max_new_tokens):
            logits, _ = self.model.forward(input_ids)
            last_token_logits = logits[0, -1, :]
            next_token_id = greedy_sample(last_token_logits)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_id.item() == self.tokenizer.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids)
        
    
    def generate(self, prompt, max_new_tokens=50):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = input_ids.to(self.model.device)

        generated_ids = input_ids
        current_input = input_ids
        past_key_values = None

        for step in range(max_new_tokens):
            logits, past_key_values = self.model.forward(current_input, past_key_values)
            last_token_logits = logits[0, -1, :]
            next_token_id = greedy_sample(last_token_logits)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            current_input = next_token_id

            if next_token_id.item() == self.tokenizer.tokenizer.eos_token_id:
                break

            final_text = self.tokenizer.decode(generated_ids)
        return final_text


"""

import time
import torch
from model.gpt2_model import GPT2Model
from tokenizer.gpt2_tokenizer import GPT2Tokenizer
from generation.sampling import greedy_sample


class InferenceEngine:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2Model(model_name)
        self.tokenizer = GPT2Tokenizer(model_name)
  
    def generate_naive(self, prompt, max_new_tokens=50, stats=None):
        start = time.perf_counter()
        input_ids = self.tokenizer.encode(prompt).to(self.model.device)
        prompt_len = input_ids.shape[1]
        new_tokens = 0

        for step in range(max_new_tokens):
            logits, _ = self.model.forward(input_ids)
            next_token_id = greedy_sample(logits[0, -1, :])
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            new_tokens += 1

            if step == 0 and stats is not None:
                torch.cuda.synchronize()
                stats["ttft"] = time.perf_counter() - start

            if next_token_id.item() == self.tokenizer.tokenizer.eos_token_id:
                break

        if stats is not None:
            torch.cuda.synchronize()
            stats["total"] = time.perf_counter() - start
            stats["new_tokens"] = new_tokens
            stats["prompt_len"] = prompt_len

        return self.tokenizer.decode(input_ids)
   
    def prefill(self, input_ids):
        logits,past_key_values = self.model.forward(input_ids, past_key_values=None)
        last_token_logits = logits[0, -1, :]
        return last_token_logits, past_key_values
    
    def decode_step(self, current_input, past_key_values):
        logits, past_key_values = self.model.forward(current_input, past_key_values)
        last_token_logits = logits[0, -1, :]
        
        return last_token_logits, past_key_values
    
    def generate(self, prompt, max_new_tokens=50, stats=None):
        start = time.perf_counter()
        input_ids = self.tokenizer.encode(prompt).to(self.model.device)
        prompt_len = input_ids.shape[1]

        # --- PREFILL: happens once ---
        last_token_logits, past_key_values = self.prefill(input_ids)
        next_token_id = greedy_sample(last_token_logits)
        generated_ids = torch.cat([input_ids, next_token_id], dim=1)
        new_tokens = 1

        if stats is not None:
            torch.cuda.synchronize()
            prefill_end = time.perf_counter()
            stats["ttft"] = prefill_end - start
            stats["prefill_time"] = prefill_end - start
            stats["decode_step_times"] = []   # <-- one entry per decode token

        current_input = next_token_id
        last_step_end = prefill_end if stats is not None else None

        # --- DECODE: repeats, one token per round ---
        for step in range(max_new_tokens - 1):
            if next_token_id.item() == self.tokenizer.tokenizer.eos_token_id:
                break

            last_token_logits, past_key_values = self.decode_step(current_input, past_key_values)
            next_token_id = greedy_sample(last_token_logits)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            current_input = next_token_id
            new_tokens += 1

            if stats is not None:
                torch.cuda.synchronize()
                now = time.perf_counter()
                stats["decode_step_times"].append(now - last_step_end)
                last_step_end = now

        if stats is not None:
            stats["total"] = last_step_end - start
            stats["decode_time"] = stats["total"] - stats["prefill_time"]
            stats["new_tokens"] = new_tokens
            stats["prompt_len"] = prompt_len

        return self.tokenizer.decode(generated_ids)

    """
    def generate(self, prompt, max_new_tokens=50, stats=None):
        start = time.perf_counter()
        input_ids = self.tokenizer.encode(prompt).to(self.model.device)
        prompt_len = input_ids.shape[1]
        generated_ids = input_ids
        current_input = input_ids
        past_key_values = None
        new_tokens = 0

        for step in range(max_new_tokens):
            logits, past_key_values = self.model.forward(current_input, past_key_values)
            next_token_id = greedy_sample(logits[0, -1, :])
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            current_input = next_token_id
            new_tokens += 1

            if step == 0 and stats is not None:
                torch.cuda.synchronize()
                stats["ttft"] = time.perf_counter() - start

            if next_token_id.item() == self.tokenizer.tokenizer.eos_token_id:
                break

        if stats is not None:
            torch.cuda.synchronize()
            stats["total"] = time.perf_counter() - start
            stats["new_tokens"] = new_tokens
            stats["prompt_len"] = prompt_len

        return self.tokenizer.decode(generated_ids)
    """