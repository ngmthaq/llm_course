from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Define constants
model_checkpoint = "gpt2"
tokenizer_checkpoint = "huggingface-course/code-search-net-tokenizer"
train_dataset_name = "huggingface-course/codeparrot-ds-train"
val_dataset_name = "huggingface-course/codeparrot-ds-valid"
output_dir = "__models/gpt2-fine-tuned"
context_length = 128

# Load the dataset
ds_train = load_dataset(train_dataset_name, split="train")
ds_valid = load_dataset(val_dataset_name, split="validation")

# Load full dataset
raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.pad_token = tokenizer.eos_token


# Tokenizer function
def tokenize_function(examples):
    outputs = tokenizer(
        examples["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


# Tokenize the dataset
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
)

# Configure the model
config = AutoConfig.from_pretrained(
    model_checkpoint,
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Load the model from config
model = GPT2LMHeadModel(config)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training arguments
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=8,
    eval_strategy="steps",
    eval_steps=5_000,
    logging_strategy="steps",
    logging_steps=5_000,
    save_strategy="steps",
    save_steps=5_000,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
)

# Trainer
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(output_dir)
