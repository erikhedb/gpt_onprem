import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Alpaca model and tokenizer
model_name = "chavinlo/alpaca-native"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# If you have a GPU available, move the model to GPU for faster inference
if torch.cuda.is_available():
    model = model.to("cuda")

# Function to generate responses from Alpaca
def chat_with_alpaca(prompt, max_length=150, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Chat loop
print("Welcome to the Alpaca Chatbot! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    alpaca_response = chat_with_alpaca(user_input)
    print(f"Alpaca: {alpaca_response}")

print("Goodbye!")
