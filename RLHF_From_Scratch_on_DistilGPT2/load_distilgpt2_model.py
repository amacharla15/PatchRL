"""
Load a causal language model from the Hugging Face Hub by name and return it
ready for inference.

The function should accept a model identifier string and produce a model object
suitable for forward passes and generation. Make sure the returned model is put
into evaluation mode so dropout and other train-only behaviors are disabled.
"""
from transformers import AutoModelForCausalLM
def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # TODO: load a causal LM by name and return it in eval mode
    # causal means model predicts tokens only from its previous tokens
    causal_LM=AutoModelForCausalLM.from_pretrained(model_name)
    causal_LM.eval()
    return causal_LM
    