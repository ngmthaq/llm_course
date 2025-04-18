from transformers import pipeline

# Create a pipeline for masked language modeling
mask_filler = pipeline(
    task="fill-mask",
    model="__models/distilbert-base-uncased-fine-tuned",
)

# Example text with a masked token
text = "This is a great [MASK]."

# Run inference on the text
preds = mask_filler(text)

# Print the predictions
for pred in preds:
    print(f">>> {pred['sequence']}")
