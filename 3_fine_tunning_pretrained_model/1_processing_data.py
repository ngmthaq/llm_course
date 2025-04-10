# Datasets: The 🤗 Datasets library provides a very simple command to download and cache a dataset on the Hub.
# In this section we will use as an example the MRPC (Microsoft Research Paraphrase Corpus) dataset,
# introduced in a paper by William B. Dolan and Chris Brockett. The dataset consists of 5,801 pairs of sentences,
# with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).
# We’ve selected it for this chapter because it’s a small dataset, so it’s easy to experiment with training on it.

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

# Define the model checkpoint
checkpoint = "bert-base-uncased"

# Load the MRPC dataset
dataset_name = "mrpc"
dataset_path = "glue"

# Load the dataset
raw_datasets = load_dataset(dataset_path, dataset_name)

# Print the dataset
print(raw_datasets)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model
model = AutoModel.from_pretrained(checkpoint)


# Tokenize the dataset callback function
def tokenize_cb(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# Tokenize the dataset
tokenized_datasets = raw_datasets.map(tokenize_cb, batched=True)

# Print the tokenized dataset
print(tokenized_datasets)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Sample a batch
samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}

[len(x) for x in samples["input_ids"]]

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

# Print the batch
print(batch)
