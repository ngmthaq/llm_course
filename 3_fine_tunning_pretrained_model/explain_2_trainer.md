# Fine-tuning BERT on the Microsoft Research Paraphrase Corpus (MRPC)

This document explains a Python script that demonstrates how to fine-tune a pre-trained BERT model on the MRPC dataset using the Hugging Face Transformers library.

## Dataset Description

The MRPC (Microsoft Research Paraphrase Corpus) dataset was introduced in a paper by William B. Dolan and Chris Brockett. It contains 5,801 pairs of sentences with labels indicating whether they are paraphrases or not (i.e., whether both sentences mean the same thing). This dataset is often used for fine-tuning experiments because of its small size, making it convenient for demonstrations and quick experiments.

## Required Libraries

```python
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
```

## Model and Dataset Setup

First, we define which pre-trained model to use and load the MRPC dataset:

```python
# Define the model checkpoint
checkpoint = "bert-base-uncased"

# Load the MRPC dataset
dataset_name = "mrpc"
dataset_path = "glue"

# Load the dataset
raw_datasets = load_dataset(dataset_path, dataset_name)
```

The code uses `bert-base-uncased`, which is a pre-trained BERT model with 12 layers and 110 million parameters. The model uses lowercase text (hence "uncased").

The dataset is loaded from the GLUE benchmark (General Language Understanding Evaluation), which includes MRPC along with other language understanding tasks.

## Tokenizer and Model

Next, we load the tokenizer and model:

```python
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

The tokenizer converts text to the format expected by the model. It splits the text into tokens, converts tokens to IDs, and handles special tokens like `[CLS]` and `[SEP]`.

We load a pre-trained BERT model with a classification head on top. Since MRPC is a binary classification task (paraphrase or not), we set `num_labels=2`.

## Data Preprocessing

We define a function to tokenize the dataset:

```python
# Tokenize the dataset callback function
def tokenize_cb(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Tokenize the dataset
tokenized_datasets = raw_datasets.map(tokenize_cb, batched=True)
```

The tokenize function takes pairs of sentences from the MRPC dataset and tokenizes them together. The `truncation=True` parameter ensures that sentences are truncated if they exceed the maximum length supported by the model (typically 512 tokens for BERT).

We apply this tokenization to the entire dataset using the `map` function, which processes the dataset in batches for efficiency.

## Data Collation

```python
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

The `DataCollatorWithPadding` handles padding of input sequences. Since sentences have different lengths, we need to pad shorter sentences to make them all the same length within a batch. This is done dynamically during training rather than ahead of time.

## Training Configuration

```python
# Training arguments
training_args = TrainingArguments(
    output_dir="__models/bert-base-uncased-fine-tuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
)
```

The `TrainingArguments` class configures the training process:

- `output_dir`: Directory where the model checkpoints will be saved
- `eval_strategy="epoch"`: Evaluate the model at the end of each epoch
- `save_strategy="epoch"`: Save model checkpoints at the end of each epoch
- `logging_strategy="epoch"`: Log training metrics at the end of each epoch

## Evaluation Metrics

```python
# Compute metrics function
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

This function computes evaluation metrics for the model:

1. It loads the appropriate metrics for MRPC from the GLUE benchmark
2. It converts model outputs (logits) to predicted classes using `argmax`
3. It computes metrics like accuracy and F1 score by comparing predictions to the true labels

The `metric.compute()` function typically returns evaluation metrics specific to the MRPC task, which include accuracy, F1 score, and others.

## Training and Evaluation

```python
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
trainer.save_model("__models/bert-base-uncased-fine-tuned")
```

The `Trainer` class simplifies the training process by handling:

- Batching
- Gradient computation and optimization
- Learning rate scheduling
- Model checkpointing
- Evaluation
- Logging

After training, we save the fine-tuned model to disk for later use.

## Summary

This code demonstrates a complete workflow for fine-tuning a pre-trained BERT model on the MRPC dataset:

1. Load the pre-trained model and dataset
2. Preprocess the data
3. Configure the training process
4. Define evaluation metrics
5. Train and evaluate the model
6. Save the fine-tuned model

This approach is an example of transfer learning, where we leverage a model pre-trained on a large corpus and adapt it to a specific downstream task with much less data than would be required to train from scratch.
