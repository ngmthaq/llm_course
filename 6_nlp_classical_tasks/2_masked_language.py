from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Define constants
model_checkpoint = "distilbert-base-uncased"
text = "This is a great [MASK]."
chunk_size = 512
dataset_name = "imdb"
output_dir = "__models/distilbert-base-uncased-fine-tuned-imdb"

# Load the IMDB dataset
imdb_dataset = load_dataset(dataset_name)

# Load the model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# Tokenize Function
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(examples["text"]))]
    return result


# Tokenize the dataset
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)


# Group texts function
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Get LM dataset
lm_dataset = tokenized_datasets.map(group_texts, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)

# Downsample the dataset
down_sampled_lm_dataset = lm_dataset["train"].train_test_split(
    train_size=10_000,
    test_size=int(0.1 * 10_000),
    seed=42,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=len(down_sampled_lm_dataset["train"]),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=down_sampled_lm_dataset["train"],
    eval_dataset=down_sampled_lm_dataset["test"],
    processing_class=tokenizer,
    compute_metrics=None,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(output_dir)
