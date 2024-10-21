# -*- coding: utf-8 -*-
"""
DialoGPT Chatbot Script
Automatically generated by Colaboratory.
Original file is located at: https://colab.research.google.com/drive/1KAg6X8RFHE0KSvFSZ__w7KGZrSqT4cZ3
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose model size
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a function for chat interactions
def chat_with_model(num_steps, generation_kwargs):
    chat_history_ids = None
    for step in range(num_steps):
        text = input(">> You: ")
        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids
        chat_history_ids = model.generate(bot_input_ids, **generation_kwargs)
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"DialoGPT: {output}")

# Greedy Search Chat
print("====Greedy Search Chat====")
generation_kwargs = {
    'max_length': 1000,
    'pad_token_id': tokenizer.eos_token_id
}
chat_with_model(num_steps=5, generation_kwargs=generation_kwargs)

# Beam Search Chat
print("====Beam Search Chat====")
generation_kwargs.update({
    'num_beams': 3,
    'early_stopping': True
})
chat_with_model(num_steps=5, generation_kwargs=generation_kwargs)

# Sampling Chat
print("====Sampling Chat====")
generation_kwargs.update({
    'do_sample': True,
    'top_k': 0
})
chat_with_model(num_steps=5, generation_kwargs=generation_kwargs)

# Sampling Chat with Temperature
print("====Sampling Chat with Temperature====")
generation_kwargs.update({
    'temperature': 0.75
})
chat_with_model(num_steps=5, generation_kwargs=generation_kwargs)

# Top-K Sampling Chat
print("====Top-K Sampling Chat====")
generation_kwargs.update({
    'top_k': 100
})
chat_with_model(num_steps=5, generation_kwargs=generation_kwargs)

# Nucleus Sampling Chat (Top-p)
print("====Nucleus Sampling Chat====")
generation_kwargs.update({
    'top_p': 0.95,
    'top_k': 0
})
chat_with_model(num_steps=5, generation_kwargs=generation_kwargs)

# Nucleus and Top-K Sampling with Multiple Responses
print("====Multiple Response Chat====")
chat_history_ids = None
for step in range(5):
    text = input(">> You: ")
    input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids is not None else input_ids
    chat_history_ids_list = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.75,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id
    )
    
    for i, chat_ids in enumerate(chat_history_ids_list):
        output = tokenizer.decode(chat_ids[bot_input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"DialoGPT {i}: {output}")
    choice_index = int(input("Choose the response you want for the next input: "))
    chat_history_ids = torch.unsqueeze(chat_history_ids_list[choice_index], dim=0)