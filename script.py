import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

# Path to the pre-trained model
live_model_path = "microsoft/DialoGPT-medium"

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(live_model_path)
model = AutoModelForCausalLM.from_pretrained(live_model_path)

# Set the pad_token to be the same as eos_token to avoid errors
tokenizer.pad_token = tokenizer.eos_token

def chat():
    print("Chatbot is ready to talk! Type 'exit' to quit.")
    
    # Initialize the chat history to None
    chat_history_ids = None

    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Encode the user input
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt", padding=True, truncation=True)

        # If there is a previous chat history, concatenate it with the new user input
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate a response from the model
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id, 
            no_repeat_ngram_size=3, 
            temperature=0.7,
            top_k=50,
            top_p=0.92
        )

        # Decode and print the response
        bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {bot_output}")

if __name__ == "__main__":
    chat()
