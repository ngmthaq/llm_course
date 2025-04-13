# Data Processing for Fine-tuning BERT

## Overview

This document explains the data processing pipeline used to prepare the MRPC dataset for fine-tuning a BERT model. The Microsoft Research Paraphrase Corpus (MRPC) consists of 5,801 sentence pairs with labels indicating whether they are paraphrases of each other.

## Code Walkthrough

### 1. Imports and Setup

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

# Define the model checkpoint
checkpoint = "bert-base-uncased"
```

We import the necessary libraries and specify the pre-trained model we'll use (`bert-base-uncased`), which is a lowercase version of BERT.

### 2. Loading the Dataset

```python
# Load the MRPC dataset
dataset_name = "mrpc"
dataset_path = "glue"

# Load the dataset
raw_datasets = load_dataset(dataset_path, dataset_name)
```

The MRPC dataset is loaded from the GLUE benchmark suite using Hugging Face's `datasets` library. This provides a convenient way to access common NLP benchmark datasets.

### 3. Loading the Model Components

```python
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model
model = AutoModel.from_pretrained(checkpoint)
```

We load both the tokenizer and model from the same checkpoint to ensure compatibility.

### 4. Tokenization Process

```python
# Tokenize the dataset callback function
def tokenize_cb(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Tokenize the dataset
tokenized_datasets = raw_datasets.map(tokenize_cb, batched=True)
```

The tokenization function processes each example by converting both sentences into token IDs. The `map` function applies this transformation to the entire dataset efficiently, with `batched=True` enabling parallel processing.

### 5. Data Collation

```python
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

The data collator handles the creation of batches from examples with varying lengths by padding sequences to a uniform length within each batch.

### 6. Sample Batch Processing

```python
# Sample a batch
samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}

[len(x) for x in samples["input_ids"]]

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

This section demonstrates how to:

1. Extract a small batch of examples
2. Remove the original text fields which are no longer needed
3. Examine the varying lengths of input sequences
4. Create a properly padded batch and inspect tensor shapes

## Key Concepts

- **Tokenization**: Converting text into tokens that the model can process
- **Padding**: Making sequences the same length for batch processing
- **Batched processing**: Efficiently transforming many examples at once
- **Data collation**: Combining individual examples into model-ready batches

This preprocessing pipeline transforms raw text data into the tensor format required for efficient training of transformer models.
