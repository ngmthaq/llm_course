from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

raw_dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint)
model = model.to(device)

training_args = SFTConfig(
    output_dir=f"__models/DeepSeek-R1-Distill-Qwen-1.5B-fine-tuned",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["test"],
)

trainer.train()

trainer.save_model(f"__models/DeepSeek-R1-Distill-Qwen-1.5B-fine-tuned")
