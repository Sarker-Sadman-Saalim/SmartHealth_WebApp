from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Training on CPU - this will be slow!")
    print("Consider installing PyTorch with CUDA support for faster training.")

# Load the synthetic Q&A dataset
qa_df = pd.read_csv("data/synthetic_qa_data.csv")

# FAST MODE: Only 500 examples for quick testing (3-5 minutes)
qa_df = qa_df.sample(n=500, random_state=42)
print(f"🚀 FAST MODE: Using {len(qa_df)} examples for quick training (~3-5 minutes)")

# Convert to Dataset format
dataset = Dataset.from_pandas(qa_df)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 doesn't have a pad token by default, so set it to eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize the dataset - FIXED VERSION
def tokenize_function(examples):
    # Combine questions and answers for each example in the batch
    texts = [q + " " + a for q, a in zip(examples['question'], examples['answer'])]
    
    # Tokenize the combined texts
    tokenized = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_tensors=None  # Return lists, not tensors
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Apply tokenization
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

print(f"Dataset size: {len(tokenized_datasets)} examples")
print(f"Sample tokenized example: {tokenized_datasets[0]}")

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Fine-tune the model - OPTIMIZED FOR 6GB GPU
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Just 1 epoch
    per_device_train_batch_size=4,  # Reduced for 6GB GPU
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Simulate batch size of 16
    warmup_steps=10,  # Minimal warmup
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,  # Log every 10 steps
    save_steps=1000,  # Don't save intermediate checkpoints
    save_total_limit=1,
    prediction_loss_only=True,
    report_to="none",
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    max_grad_norm=1.0,  # Gradient clipping
)

print(f"⚡ Training optimized for your 6GB GPU...")
print(f"   Effective batch size: 16 (4 x 4 gradient accumulation)")
print(f"   Total steps: ~{len(dataset) // 4} steps")
print(f"   Estimated time: 5-7 minutes")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# Save the fine-tuned model and tokenizer
print("Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_obesity_model")
tokenizer.save_pretrained("./fine_tuned_obesity_model")

print("✅ Fine-tuning complete! Model saved to './fine_tuned_obesity_model'")