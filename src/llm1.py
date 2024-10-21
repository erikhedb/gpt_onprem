import os
import torch
import GPUtil

try:
    import deepspeed
    deepspeed_available = True
except ImportError:
    deepspeed_available = False
    print("DeepSpeed is not available. Running without DeepSpeed optimizations.")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure compatibility with the latest pytree registration and secure model loading
torch.utils._pytree.register_pytree_node  # Updated to avoid deprecated warning

def load_model(model_name="EleutherAI/gpt-j-6B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize DeepSpeed for efficient inference, if available
    if deepspeed_available:
        try:
            model, _ = deepspeed.init_inference(
                model, mp_size=1, dtype=torch.float16, replace_method='auto', replace_with_kernel_inject=True
            )
            print("DeepSpeed initialized for efficient inference.")
        except Exception as e:
            print(f"DeepSpeed initialization failed: {e}. Proceeding without DeepSpeed.")
    
    return tokenizer, model

def generate_textOLD(tokenizer, model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def generate_text(tokenizer, model, prompt, max_length=100, num_beams=5, top_p=0.9, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.strip()


def print_gpu_stats():
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
            print(f"  Memory Total: {gpu.memoryTotal}MB")
            print(f"  Memory Used: {gpu.memoryUsed}MB")
            print(f"  Memory Free: {gpu.memoryFree}MB")
            print(f"  GPU Load: {gpu.load * 100:.1f}%")
    else:
        print("CUDA not available. No GPU statistics to display.")

def print_cuda_cpu_info():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Using CUDA Device: {torch.cuda.get_device_name(current_device)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(current_device) / 1024 ** 2:.2f}MB")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved(current_device) / 1024 ** 2:.2f}MB")
    else:
        print("CUDA not available. Running on CPU.")

if __name__ == "__main__":
    # Menu to choose the model
    while True:
        print("Select a model to load:")
        print("1. EleutherAI/gpt-j-6B")
        print("2. gpt2")
        print("3. EleutherAI/gpt-neo-125M")
        print("Type 'exit' to exit.")
        user_choice = input("Enter the number of the model you want to use: ").strip()

        if user_choice.lower() == 'exit':
            print("Exiting...")
            break
        elif user_choice == '1':
            model_name = "EleutherAI/gpt-j-6B"
        elif user_choice == '2':
            model_name = "gpt2"
        elif user_choice == '3':
            model_name = "EleutherAI/gpt-neo-125M"
        else:
            print("Invalid choice. Please try again.")
            continue

        # Load the chosen model
        tokenizer, model = load_model(model_name)

        # Print GPU statistics
        print_gpu_stats()
        # Print CUDA or CPU information
        print_cuda_cpu_info()

        # Infinite loop for LLM dialog
        while True:
            prompt = input("You: ")
            if prompt.lower() == 'exit':
                print("Exiting...")
                break
            output = generate_text(tokenizer, model, prompt)
            print("AI: ", output.strip())

        # Exit the outer loop if user wants to quit
        if prompt.lower() == 'exit':
            break
