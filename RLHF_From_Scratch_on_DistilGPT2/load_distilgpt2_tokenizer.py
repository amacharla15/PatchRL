#Implement load_distilgpt2_tokenizer which loads the tokenizer for a 
#distilgpt2-compatible model from the Hugging Face Hub by name. 
#The returned object should be a standard Hugging Face tokenizer instance 
#whose vocab_size matches the underlying checkpoint and that can encode and 
#decode text out of the box. Default the model_name argument to a tiny stand-in 
#(sshleifer/tiny-gpt2) so it runs quickly on CPU during tests.

from transformers import AutoTokenizer
def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # TODO: load and return the Hugging Face tokenizer for the given model name.
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    return tokenizer
