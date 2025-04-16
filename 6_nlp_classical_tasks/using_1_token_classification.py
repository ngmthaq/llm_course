from transformers import pipeline

# Using Model
token_classifier = pipeline(
    "token-classification",
    model="__models/bert-base-cased-fine-tuned",
    aggregation_strategy="simple",
)

result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

print(result)
