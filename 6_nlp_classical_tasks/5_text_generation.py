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

context_length = 128

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer"
)


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


tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
)

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="6_nlp_classical_tasks/_5_text_generation",
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

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()

trainer.save_model("6_nlp_classical_tasks/_5_text_generation")
