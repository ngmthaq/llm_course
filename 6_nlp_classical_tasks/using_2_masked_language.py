from transformers import pipeline

mask_filler = pipeline(
    task="fill-mask",
    model="6_nlp_classical_tasks/_2_masked_language_trainer",
)

text = "This is a great [MASK]."

preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
