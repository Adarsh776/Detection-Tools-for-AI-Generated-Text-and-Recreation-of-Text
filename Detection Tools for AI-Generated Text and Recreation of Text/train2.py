import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd

# Load the training essays
train_essays = pd.read_csv('train_essays.csv')

# Save the text column to a text file for training
train_essays['text'].to_csv('train_texts.txt', header=False, index=False)

# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load and preprocess the dataset
def preprocess_data(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

# Function to train the model
def train_gpt2(train_file, output_dir):
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Preprocess the data
    train_dataset = preprocess_data(train_file, tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Path to the training data text file
train_file = 'train_texts.txt'
output_dir = 'fine_tuned_gpt2'

# Train the model
train_gpt2(train_file, output_dir)

print("Model fine-tuning completed and saved at:", output_dir)
