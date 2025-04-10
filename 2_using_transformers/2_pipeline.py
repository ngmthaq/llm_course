from transformers import pipeline

# Load the sentiment-analysis pipeline
# This will automatically download the model and tokenizer for you
# The pipeline will use the default model for sentiment analysis
# You can specify a different model by passing the `model` argument
# The pipeline will also automatically handle the input and output for you
classifier = pipeline("sentiment-analysis")

# Run inference on a list of sequences
output = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

print(output)
