import os
import subprocess
import sys
import torch
from google import protobuf

from transformers import AutoModelForCausalLM, AutoTokenizer

def reset_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA memory cache cleared.")

def load_model(model_name="chavinlo/alpaca-native"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
    try:
        # Clear cache before loading the model to avoid fragmentation issues
        reset_cuda_memory()
        # Attempt to load the model in fp16
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, cache_dir="./model_cache", low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            model = model.cuda()
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Loading model on CPU instead.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, cache_dir="./model_cache", low_cpu_mem_usage=True
        )
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_length=50, num_beams=3, top_p=0.8, temperature=0.7, repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() and next(model.parameters()).is_cuda else 'cpu')
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.strip()

def generate_batch_text(tokenizer, model, prompts, max_length=100):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in output]

def print_gpu_stats():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}")
        print(f"  Memory Total: {gpu.total_memory / 1024 ** 2:.2f}MB")
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
        print(f"  Cached Memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f}MB")
    else:
        print("CUDA not available. No GPU statistics to display.")

if __name__ == "__main__":
    # Load Alpaca model
    tokenizer, model = load_model("chavinlo/alpaca-native")

    # Print GPU statistics
    print_gpu_stats()

    # Infinite loop for LLM dialog
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        output = generate_text(tokenizer, model, prompt)
        print("AI: ", output.strip())
