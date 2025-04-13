from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "3_fine_tunning_pretrained_model/_test_trainer"
sentence1 = "The weather is nice today."
sentence2 = "It's a beautiful day outside."

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

paraphrase_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = paraphrase_pipeline(f"{sentence1} [SEP] {sentence2}")
print(result)
