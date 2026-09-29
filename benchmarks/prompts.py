def make_prompt(engine, target_tokens, unit="hello "):
    text = ""
    while True:
        text += unit
        token_count = engine.tokenizer.encode(text).shape[1]
        if token_count >= target_tokens:
            return text