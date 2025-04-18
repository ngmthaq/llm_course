from transformers import pipeline

# Load the token classification pipeline
token_classifier = pipeline(
    "token-classification",
    model="__models/bert-base-cased-fine-tuned",
    aggregation_strategy="simple",
)

# Run inference on a sequence
result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# Print the result
print(result)
