from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import Trainer
import numpy as np
import evaluate

# Define constants
checkpoint = "Helsinki-NLP/opus-mt-en-fr"
output_dir = "__models/opus-mt-en-fr-fine-tuned"
dataset_name = "kde4"
dataset_config_name = "en-fr"

# Load dataset
raw_datasets = load_dataset(
    path=dataset_name,
    trust_remote_code=True,
    lang1="en",
    lang2="fr",
)

# Downsample the dataset
split_datasets = raw_datasets["train"].train_test_split(
    train_size=0.2,  # increase train size to higher accuracy
    test_size=0.1,  # increase test size to higher accuracy
    seed=42,
)

# Rename the test set to validation
split_datasets["validation"] = split_datasets.pop("test")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt")


# Tokenize function
def preprocess_function(examples):
    inputs = [example["en"] for example in examples["translation"]]
    targets = [example["fr"] for example in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=128, truncation=True)


# Tokenize the dataset
tokenizer_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Data collator
data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)

# Load the metric
metric = evaluate.load("sacrebleu")


# Compute metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds[0] if isinstance(preds, tuple) else preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [label.strip() for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="__models/distilbert-base-uncased-fine-tuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    # Below option to fix mps out of memory error
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=False,
    dataloader_num_workers=0,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenizer_datasets["train"],
    eval_dataset=tokenizer_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the model
trainer.save_model("__models/distilbert-base-uncased-fine-tuned")
