

from transformers import GPT2Tokenizer as HFGPT2Tokenizer


class GPT2Tokenizer:
    def __init__(self,model_name="gpt2"):
        self.tokenizer =  HFGPT2Tokenizer.from_pretrained(model_name)

    def encode(self, text):
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"]
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

