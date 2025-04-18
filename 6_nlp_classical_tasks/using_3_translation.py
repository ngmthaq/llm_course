from transformers import pipeline

# Load the translation pipeline
model_checkpoint = "__models/distilbert-base-uncased-fine-tuned"
translator = pipeline("translation", model=model_checkpoint)
result = translator("Default to expanded threads")

# Print the result
print(result)
