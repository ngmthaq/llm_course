# Token classification is a generic task encompasses any problem that can be formulated as
# "attributing a label to each token in a sequence of tokens".

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline

# Define constants
checkpoint = "bert-base-cased"
output_dir = "__models/bert-base-cased-fine-tuned"

# Load the CoNLL2003 dataset
raw_datasets = load_dataset("conll2003", trust_remote_code=True)
print(raw_datasets)

# NER features
# NER: Named Entity Recognition
ner_features = raw_datasets["train"].features["ner_tags"]
print(ner_features)

# Label names
# O means the word doesn’t correspond to any entity.
# B-PER/I-PER means the word corresponds to the beginning of/is inside a person entity.
# B-ORG/I-ORG means the word corresponds to the beginning of/is inside an organization entity.
# B-LOC/I-LOC means the word corresponds to the beginning of/is inside a location entity.
# B-MISC/I-MISC means the word corresponds to the beginning of/is inside a miscellaneous entity.
label_names = ner_features.feature.names
print(label_names)

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

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(tokenizer.is_fast)


# Grant special tokens to -100
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


# Tokenize dataset callback
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


# Tokenize the dataset
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Compute metrics function
metric = evaluate.load("seqeval")


# Compute metrics function
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# Mapping id to label and label to id
id2label = {index: label for index, label in enumerate(label_names)}
label2id = {label: index for index, label in id2label.items()}

# Model
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# Training arguments
args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
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
trainer.save_model(output_dir)
