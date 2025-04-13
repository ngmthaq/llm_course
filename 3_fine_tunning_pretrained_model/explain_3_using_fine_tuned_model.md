# Using a Fine-Tuned Model for Paraphrase Detection

This document explains how to use a fine-tuned model for paraphrase detection tasks.

## Code Explanation

The code demonstrates how to load and use a fine-tuned model to determine if two sentences are paraphrases of each other:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "3_fine_tunning_pretrained_model/_test_trainer"
sentence1 = "The weather is nice today."
sentence2 = "It's a beautiful day outside."

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

paraphrase_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = paraphrase_pipeline(f"{sentence1} [SEP] {sentence2}")
print(result)
```

## Step-by-Step Explanation

1. **Import Required Libraries**:

   - `AutoTokenizer`: Loads the tokenizer that was used during fine-tuning
   - `AutoModelForSequenceClassification`: Loads the classification model
   - `pipeline`: A high-level API for easy model inference

2. **Define Inputs**:

   - `model_path`: Path to the directory containing the fine-tuned model
   - `sentence1` and `sentence2`: The pair of sentences to compare

3. **Load the Fine-Tuned Model and Tokenizer**:

   - The model and tokenizer are loaded from the specified local directory
   - This model was previously fine-tuned for the paraphrase detection task

4. **Create a Pipeline for Inference**:

   - A text classification pipeline is created using the loaded model and tokenizer
   - This simplifies the process of tokenizing inputs and interpreting outputs

5. **Run Inference**:

   - The sentences are combined with a `[SEP]` token between them
   - This follows the format expected by the model during training
   - The pipeline handles tokenization and model inference

6. **Output the Results**:
   - The result contains the classification label (e.g., "PARAPHRASE" or "NOT_PARAPHRASE")
   - It also includes the confidence score for the prediction

## Usage Notes

- The model expects two sentences separated by the `[SEP]` token
- The output classification labels depend on how the model was fine-tuned
- Higher confidence scores indicate stronger model certainty
