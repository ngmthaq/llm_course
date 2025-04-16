# Named Entity Recognition (NER) with Transformers

This code demonstrates how to perform token classification for Named Entity Recognition (NER) using the Hugging Face transformers library. The example uses the CoNLL2003 dataset and fine-tunes a BERT model to identify and classify entities in text.

## What is Token Classification?

Token classification refers to tasks where we assign a label to each token in a sequence. Named Entity Recognition is a common token classification task where we identify entities like people, organizations, locations, etc. in text.
Unlike text classification which assigns a single label to an entire text sequence, token classification operates at the token level, requiring more fine-grained predictions. This makes it particularly useful for extracting structured information from unstructured text.

## Code Walkthrough

### 1. Setting Up Dependencies

```python
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
```

These imports provide the essential components:

- evaluate: Hugging Face's library for model evaluation metrics
- datasets: Simplifies loading and processing datasets
- transformers: Contains model architectures, tokenizers, and training utilities
- DataCollatorForTokenClassification: Specially designed for token classification tasks, handling padding and masking appropriately

### 2. Loading the Dataset

The CoNLL2003 dataset is a standard benchmark for NER tasks:

```python
# Load the CoNLL2003 dataset
raw_datasets = load_dataset("conll2003", trust_remote_code=True)
print(raw_datasets)

# Examine the NER features
ner_features = raw_datasets["train"].features["ner_tags"]
print(ner_features)
```

The CoNLL2003 dataset contains sentences split into tokens, with corresponding part-of-speech (POS) tags, syntactic chunk tags, and named entity tags. For NER, we focus on the "ner_tags" column, which contains numerical encodings of entity labels for each token.

### 3. Understanding the Labels

The NER tags in CoNLL2003 follow the IOB (Inside-Outside-Beginning) format:

- **O**: Token doesn't belong to any entity
- **B-PER/I-PER**: Beginning of/Inside a person entity
- **B-ORG/I-ORG**: Beginning of/Inside an organization entity
- **B-LOC/I-LOC**: Beginning of/Inside a location entity
- **B-MISC/I-MISC**: Beginning of/Inside a miscellaneous entity

The "B-" prefix marks the beginning of an entity, while "I-" marks tokens inside an entity. This distinction is crucial for properly identifying entity boundaries, especially when multiple entities of the same type appear adjacent to each other.

```python
# Get the label names
label_names = ner_features.feature.names
print(label_names)
# Output: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### 4. Visualizing the Dataset

This code aligns words with their corresponding NER tags for better visualization, making it easier to understand how entities are labeled in context:

```python
# Test the dataset
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
# Output would show tokens aligned with their NER labels
# Example:
# EU   rejects German call to boycott British lamb .
# B-ORG O       B-MISC  O    O  O       B-MISC   O   O
```

This visualization helps us quickly see how entities are distributed in the text and verify that the labels are correctly aligned with the corresponding tokens.

### 5. Setting Up the Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer.is_fast)
# Output: True (indicates we're using the fast Rust-based implementation)
```

We use the BERT tokenizer in its case-sensitive variant ("bert-base-cased"), which is important for NER since capitalization is often a crucial signal for identifying proper nouns and entities. The "fast" tokenizer implementation provides methods like word_ids() that are essential for aligning tokens with their original words.

### 6. Aligning Labels with Tokens

A critical step in token classification is aligning the original labels with tokenized inputs. This is necessary because tokenizers may split words into multiple tokens:

```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels
```

This function handles several important aspects of label alignment:

1. Maps special tokens (like [CLS], [SEP]) to -100, which is ignored in the loss calculation
2. Maintains the original label for the first token of each word
3. For subword tokens, properly converts B-XXX tags to I-XXX tags
4. Ensures each token gets exactly one label

For example, if "playing" is tokenized as ["play", "##ing"] with an original label of "B-MISC", the function will assign ["B-MISC", "I-MISC"] to ensure correct entity boundaries.

### 7. Tokenizing the Dataset

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# Apply tokenization to the entire dataset
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

This function:

1. Tokenizes each example, specifying is_split_into_words=True since our input is already pre-tokenized into words
2. Gets the mapping between tokens and their original words using word_ids()
3. Aligns the labels with the tokenized inputs using our custom function
4. Applies this processing to the entire dataset using the batched map operation

The resulting dataset contains tokenized inputs and properly aligned labels, ready for model training. The remove_columns parameter ensures we only keep the columns needed for training.

### 8. Preparing for Training

```python
# Data collator handles batching
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Load evaluation metric
metric = evaluate.load("seqeval")
```

The DataCollatorForTokenClassification is a specialized data collator that:

1. Pads sequences to the same length within a batch
2. Creates attention masks to distinguish real tokens from padding
3. Properly handles the label padding (using -100)

The "seqeval" metric is specifically designed for sequence labeling tasks like NER. Unlike token-level metrics, it evaluates entity-level performance by considering entire entity spans rather than individual tokens, which better reflects the model's practical utility.

### 9. Setting Up Evaluation Metrics

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Filter out tokens with label -100 (padding/special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate metrics using seqeval
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

This function:

1. Converts model logits to predicted label indices using argmax
2. Filters out padding and special tokens (labeled as -100)
3. Converts numerical labels to their string representations (e.g., "B-PER")
4. Calculates key metrics:
   - Precision: Proportion of predicted entities that are correct
   - Recall: Proportion of actual entities that were found
   - F1 Score: Harmonic mean of precision and recall
   - Accuracy: Overall token classification accuracy

Seqeval calculates these metrics at the entity level rather than token level, which gives a better picture of the model's practical performance in identifying complete entities.

### 10. Initializing the Model

```python
# Create id-to-label and label-to-id mappings
id2label = {index: label for index, label in enumerate(label_names)}
label2id = {label: index for index, label in id2label.items()}

# Initialize the model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    id2label=id2label,
    label2id=label2id,
)
```

Here we:

- Create mappings between numeric IDs and string labels for the model
- Initialize a BERT model specifically configured for token classification
- Use the pre-trained "bert-base-cased" checkpoint as our starting point

The AutoModelForTokenClassification class adds a token classification head (a linear layer) on top of the pre-trained BERT model. This head will output logits for each class for each token position, which will be converted to probabilities during inference.

### 11. Training Setup and Execution

```python
# Training arguments
args = TrainingArguments(
    output_dir="__models/bert-base-cased-fine-tuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("__models/bert-base-cased-fine-tuned")
```

The training configuration:

1. Uses a learning rate of 2e-5, which is a common value for fine-tuning pre-trained transformers
2. Trains for 3 epochs - more might lead to overfitting, fewer might not capture enough patterns
3. Applies weight decay for regularization
4. Evaluates and saves the model after each epoch to track progress

The Trainer class orchestrates the entire training process, handling:

- Batching and optimization
- Learning rate scheduling
- Gradient accumulation
- Evaluation
- Model checkpointing
- Early stopping (if configured)

## Key Concepts

- **Token Classification**: Assigning labels to individual tokens in text
- **Named Entity Recognition**: Identifying and classifying entities in text
- **IOB Format**: Inside-Outside-Beginning tagging scheme for entities
- **Token Alignment**: Matching original labels to tokenized words
- **Evaluation Metrics**: Using seqeval for NER evaluation (precision, recall, F1)

This code demonstrates a complete pipeline for fine-tuning a pre-trained transformer model (BERT) for Named Entity Recognition, from data preparation to training and evaluation.
