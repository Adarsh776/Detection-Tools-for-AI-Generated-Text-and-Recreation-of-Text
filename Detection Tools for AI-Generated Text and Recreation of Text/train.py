import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def tokenize_input(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    return inputs

def calculate_perplexity(text):
    inputs = tokenize_input(text)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.item()

def is_generated_by_gpt(text, threshold=50):
    perplexity = calculate_perplexity(text)
    if perplexity < threshold:
        return True
    else:
        return False

# Example usage
text = """ Hi my name is shrryanth hg wt is youyr name 
"""
if is_generated_by_gpt(text):
    print("This text is likely generated by GPT.")
else:
    print("This text is likely human-generated.")
