# Load model directly
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

huggingface_hub.login("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", add_to_git_credential=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                          token='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', cache_dir='../../model')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                             token='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', cache_dir='../../model')
