import torch


def build_batch(engine, prompts):
    tok = engine.tokenizer.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # TODO 1: encode each prompt, get 1D token tensors
    token_lists = [engine.tokenizer.encode(p)[0] for p in prompts]

    # TODO 2: find the longest sequence
    max_len = max(t.shape[0] for t in token_lists)

    padded_rows = []
    mask_rows = []

    for t in token_lists:
        pad_len = max_len - t.shape[0]

        # TODO 3: build pad tokens and the real attention mask, LEFT-padded
        pad_tokens = torch.full((pad_len,), tok.pad_token_id, dtype=t.dtype)
        row = torch.cat([pad_tokens, t])

        mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                           torch.ones(t.shape[0], dtype=torch.long)])

        padded_rows.append(row)
        mask_rows.append(mask)

    # TODO 4: stack into (batch_size, max_len) tensors
    input_ids = torch.stack(padded_rows)
    attention_mask = torch.stack(mask_rows)

    return input_ids, attention_mask