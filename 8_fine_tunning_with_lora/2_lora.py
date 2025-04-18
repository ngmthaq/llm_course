from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
import torch

# Define the constants
dataset_name = "HuggingFaceTB/smoltalk"
dataset_config_name = "all"
checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
output_dir = "__models/DeepSeek-R1-Distill-Qwen-1.5B-lora-fine-tuned"

# Lora configuration options
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_task_type = TaskType.CAUSAL_LM
lora_target_modules = ["q_proj", "v_proj"]

# Set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the dataset
raw_dataset = load_dataset(dataset_name, dataset_config_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load lora model
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    task_type=lora_task_type,
    target_modules=lora_target_modules,
)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
lora_model = get_peft_model(model, lora_config)
lora_model = lora_model.to(device)

# Load training arguments
training_args = SFTConfig(
    output_dir=output_dir,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
)

# Train the model
trainer = SFTTrainer(
    model=lora_model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["test"],
    peft_config=lora_config,
)

# Start training
trainer.train()

# Save the model
trainer.save_model(output_dir)
