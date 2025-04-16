from transformers import pipeline

model_checkpoint = "__models/distilbert-base-uncased-fine-tuned"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
