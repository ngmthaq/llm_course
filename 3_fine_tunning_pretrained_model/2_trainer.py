# Datasets: The 🤗 Datasets library provides a very simple command to download and cache a dataset on the Hub.
# In this section we will use as an example the MRPC (Microsoft Research Paraphrase Corpus) dataset,
# introduced in a paper by William B. Dolan and Chris Brockett. The dataset consists of 5,801 pairs of sentences,
# with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).
# We’ve selected it for this chapter because it’s a small dataset, so it’s easy to experiment with training on it.

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# Define the model checkpoint, model output directory
checkpoint = "bert-base-uncased"
output_dir = "__models/bert-base-uncased-fine-tuned"

# Load the MRPC dataset
dataset_name = "mrpc"
dataset_path = "glue"

# Load the dataset
raw_datasets = load_dataset(dataset_path, dataset_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


# Tokenize the dataset callback function
def tokenize_cb(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# Tokenize the dataset
tokenized_datasets = raw_datasets.map(tokenize_cb, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
)


# Compute metrics function
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(output_dir)
