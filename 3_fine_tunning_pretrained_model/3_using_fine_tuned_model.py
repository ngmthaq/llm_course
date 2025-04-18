from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Define the model path and the sentences to be compared
model_path = "__models/bert-base-uncased-fine-tuned"
sentence1 = "The weather is nice today."
sentence2 = "It's a beautiful day outside."

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline for paraphrase classification
paraphrase_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = paraphrase_pipeline(f"{sentence1} [SEP] {sentence2}")
print(result)
