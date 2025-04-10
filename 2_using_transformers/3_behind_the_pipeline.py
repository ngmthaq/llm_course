from transformers import AutoTokenizer, AutoModel

# Define the model checkpoint and the sequences to be processed
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Load the tokenizer and convert the sequences to input IDs
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Load the model and perform inference
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**model_inputs)

# Print the outputs
print(outputs)
