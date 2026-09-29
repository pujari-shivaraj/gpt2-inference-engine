from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Load the tokenizer using the "gpt2" checkpoint
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the model using the "gpt2" checkpoint
model = GPT2LMHeadModel.from_pretrained("gpt2")

model = model.to(device)

# Print the tokenizer's vocabulary size
print(tokenizer.vocab_size)

# Print the type of the model object
print(type(model))

prompt = "The future of AI"

encoded = tokenizer(prompt, return_tensors="pt")

encoded = tokenizer(
    prompt,
    return_tensors="pt"
)

encoded = {
    key: value.to(device)
    for key, value in encoded.items()
}

# TODO 2: print the whole encoded object
print(encoded)

# TODO 3: print just the input_ids tensor
print(encoded["input_ids"])

# TODO 4: print the shape of input_ids
print(encoded["input_ids"].shape)

print(tokenizer.tokenize("unbelievably"))

"""

with torch.no_grad():
    outputs = model(encoded["input_ids"])

logits = outputs.logits
print(logits.shape)

last_token_logits = logits[0, -1, :]
print(last_token_logits.shape)

next_token_id = torch.argmax(last_token_logits)
print(next_token_id)

next_token = tokenizer.decode([next_token_id])
print(next_token)


"""



generated_ids = encoded["input_ids"]

for step in range(6):
    with torch.no_grad():
        outputs = model(generated_ids)

    logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(last_token_logits)

    # reshape scalar -> (1,1) so it can be concatenated
    next_token_id = next_token_id.view(1, 1)

    # TODO: concatenate next_token_id onto generated_ids along dim=1
    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

# TODO: decode the FULL generated_ids sequence back to text
final_text = tokenizer.decode(generated_ids[0])
print(final_text)