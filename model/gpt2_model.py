import torch
from transformers import GPT2LMHeadModel


class GPT2Model:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        # Evaluation mode
        self.model.eval()
    
    """
    def forward(self, input_ids, past_key_values=None):
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
    

        return outputs.logits, outputs.past_key_values
    """
    def forward(self, input_ids, past_key_values=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )
        return outputs.logits, outputs.past_key_values