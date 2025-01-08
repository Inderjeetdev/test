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
    print("Chatbot is ready to talk! Simulated conversation begins...")

    # Simulated inputs for GitHub Actions
    simulated_inputs = ["hello", "how are you", "what is your name", "exit"]

    # Initialize the chat history to None
    chat_history_ids = None

    for user_input in simulated_inputs:
        print(f"You: {user_input}")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Encode the user input with attention mask
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        attention_mask = new_user_input_ids.ne(tokenizer.pad_token_id).long()
        
        # If there is a previous chat history, concatenate it with the new user input
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=-1)  # Concatenate attention masks
        else:
            bot_input_ids = new_user_input_ids
        
        # Generate a response from the model
        chat_history_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,  # Pass the attention mask explicitly
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
