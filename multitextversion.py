import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import GPTSw3Tokenizer
from torch.utils.data import Dataset, DataLoader
import re
import torch.nn.functional as F
failedgenerations = 0
epochs = 5

# Load pre-trained model and tokenizer
model_name = "GPT-4/GPT-3"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare chat data
class ChatDataset(Dataset):
    def __init__(self, user_inputs, responses, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set the padding token
        for user_input, response in zip(user_inputs, responses):
            # Combine user input and response into a single conversation
            conversation = f"User: {user_input} Response: {response}"
            # Encode the conversation
            encodings_dict = tokenizer(conversation, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# Read user inputs and responses from files
with open('user_inputs.txt', 'r', encoding='utf-8') as user_inputs_file, open('responses.txt', 'r', encoding='utf-8') as responses_file:
    user_inputs = [line.strip() for line in user_inputs_file]
    responses = [line.strip() for line in responses_file]

# Create dataset and dataloader
max_length = 128
batch_size = 8
dataset = ChatDataset(user_inputs, responses, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fine-tuning the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-4)
total_steps = len(dataloader) * epochs  # Number of epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def trainepochs(epochs):
    NewBest = float('inf')
    for epoch in range(epochs):  # Number of epochs
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.item())
        print(f"Epoch {epoch + 1}")
        if loss_value < NewBest:
            NewBest = loss_value
            print(f"Loss: {loss.item()}")

trainepochs(epochs)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")

def generate_response(prompt, model, tokenizer, failedgenerations, max_length=250):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate response with adjusted parameters
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Adjust temperature for randomness
        top_k=2,  # Limit words
        do_sample=True,  # Enable sampling-based generation
        eos_token_id=tokenizer.eos_token_id  # Stop at end-of-sentence token
    )

    # Decode and return the response
    prompts = prompt
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Decode and return the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    response = response.replace("response", "").replace("Response", "").strip()
    response = re.sub(r'^[^a-zA-Z]*', '', response)
    response = response.split(':')[0].strip()
    if prompt == response and failedgenerations < 25:
        failedgenerations += 1
    else:
        print(f"{failedgenerations}")
        return response

def chat_with_model(model, tokenizer):
    model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")
    model.to(device)
    print("Chat with the model! Type 'quit' to exit.")
    while True:

        # Get user input
        user_input = input("You: ")
        with open('user_inputs.txt', 'a', encoding='utf-8') as file:
            file.write(f"\n{user_input}")

        # Check if the user wants to quit
        if user_input.lower() == 'quit':
            model.save_pretrained("fine_tuned_model")
            break

        # Generate a response
        response = generate_response(user_input, model, tokenizer, 0)

        # Print the response
        print(f"{response}")
        wasgoodmaybe = input('wasgood?')
        if wasgoodmaybe == "yes" or wasgoodmaybe == "y":
            with open('responses.txt', 'a', encoding='utf-8') as file:
                file.write(f"\n{response}")
        elif wasgoodmaybe == "no" or wasgoodmaybe == "n":
            with open('responses.txt', 'a', encoding='utf-8') as file:
                file.write(f"\n{input('Expected Output:')}")
        else:
            with open('responses.txt', 'a', encoding='utf-8') as file:
                file.write(f"\n{response}")
        trainepochs(1)

chat_with_model(model, tokenizer)