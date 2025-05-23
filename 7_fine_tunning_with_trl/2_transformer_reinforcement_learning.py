from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the constants
checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_name = "HuggingFaceTB/smoltalk"
dataset_config_name = "all"
output_dir = "__models/DeepSeek-R1-Distill-Qwen-1.5B-fine-tuned"

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

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint)
model = model.to(device)

# Training arguments
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
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["test"],
)

# Start training
trainer.train()

# Save the model
trainer.save_model(output_dir)
