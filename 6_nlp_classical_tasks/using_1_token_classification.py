from transformers import pipeline

# Using Model
token_classifier = pipeline(
    "token-classification",
    model="6_nlp_classical_tasks/_1_token_classification_trainer",
    aggregation_strategy="simple",
)

result = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

print(result)
