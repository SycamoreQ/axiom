#!/usr/bin/env python3
import json

from gguf import GGUFReader

reader = GGUFReader("testdata/Llama-3.2-1B-Instruct-Q4_0.gguf")

# Extract tokenizer data
tokens = reader.fields["tokenizer.ggml.tokens"]
token_types = reader.fields["tokenizer.ggml.token_type"]
scores = reader.fields.get("tokenizer.ggml.scores", None)
merges = reader.fields.get("tokenizer.ggml.merges", None)
bos_id = reader.fields.get("tokenizer.ggml.bos_token_id", None)
eos_id = reader.fields.get("tokenizer.ggml.eos_token_id", None)

# Convert to proper JSON format that matches HuggingFace tokenizer.json
vocab = {token: idx for idx, token in enumerate(tokens)}

tokenizer_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"String": " "}, "behavior": "Removed"},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        ],
    },
    "post_processor": None,
    "decoder": {"type": "ByteLevel", "add_prefix_space": False},
    "model": {
        "type": "BPE",
        "vocab": vocab,
        "merges": merges if merges else [],
        "ignore_merges": False,
        "byte_fallback": True,
    },
}

# Save to file
with open("testdata/tokenizer_fixed.json", "w") as f:
    json.dump(tokenizer_json, f, indent=2)

print(f"✅ Saved tokenizer with {len(tokens)} tokens")
print(f"   BOS token ID: {bos_id}")
print(f"   EOS token ID: {eos_id}")
print(f"   First 5 tokens: {list(vocab.keys())[:5]}")
