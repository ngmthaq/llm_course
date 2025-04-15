from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import Trainer
import numpy as np
import evaluate

raw_datasets = load_dataset(path="kde4", trust_remote_code=True, lang1="en", lang2="fr")

split_datasets = raw_datasets["train"].train_test_split(
    train_size=0.2,  # increase train size to higher accuracy
    test_size=0.1,  # increase test size to higher accuracy
    seed=42,
)

split_datasets["validation"] = split_datasets.pop("test")

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")


def preprocess_function(examples):
    inputs = [example["en"] for example in examples["translation"]]
    targets = [example["fr"] for example in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=128, truncation=True)


tokenizer_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)

metric = evaluate.load("sacrebleu")


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


args = Seq2SeqTrainingArguments(
    output_dir="6_nlp_classical_tasks/_3_translation_trainer",
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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenizer_datasets["train"],
    eval_dataset=tokenizer_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("6_nlp_classical_tasks/_3_translation_trainer")
