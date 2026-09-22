"""Model loading and text generation (greedy / top-p) for GPT-2 Small"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import MODEL_NAME, TOP_P, GEN_LENGTH

_tokenizer = None
_model = None
_device = None


def load_model():
    """Lazy-load the model once and reuse across all generations."""
    global _tokenizer, _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_NAME} on {_device} ...")
        _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def generate(prompt_text, strategy, seed=None):
    """Generates GEN_LENGTH new tokens continuing prompt_text.
    strategy: "greedy" or "top_p" (seed required for "top_p").
    Returns: (continuation_token_ids, continuation_text)."""
    tokenizer, model, device = load_model()
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    gen_kwargs = dict(max_new_tokens=GEN_LENGTH, pad_token_id=tokenizer.eos_token_id)
    if strategy == "greedy":
        gen_kwargs.update(do_sample=False)
    elif strategy == "top_p":
        if seed is None:
            raise ValueError("top_p generation requires a seed")
        torch.manual_seed(seed)
        gen_kwargs.update(do_sample=True, top_p=TOP_P, top_k=0, temperature=1.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    cont_ids = output_ids[0][input_ids.shape[1]:].tolist()
    cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
    return cont_ids, cont_text
