from transformers import pipeline

mask_filler = pipeline(
    task="fill-mask",
    model="__models/distilbert-base-uncased-fine-tuned",
)

text = "This is a great [MASK]."

preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
