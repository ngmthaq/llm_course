## TRANSFORMERS

- The 🤗 Transformers library was created to solve this problem. Its goal is to provide a single API
  through which any Transformer model can be loaded, trained, and saved. The library's main features are:
  - Ease of use: Downloading, loading, and using a state-of-the-art NLP model for inference can be done in just two lines of code.
  - Flexibility: At their core, all models are simple PyTorch nn.Module or TensorFlow tf.keras.Model classes and can be handled like
    any other models in their respective machine learning (ML) frameworks.
  - Simplicity: Hardly any abstractions are made across the library. The “All in one file” is a core concept: a model's forward pass
    is entirely defined in a single file, so that the code itself is understandable and hackable.

## PIPELINES

- The pipeline() function is the first tool from the 🤗 Transformers library. It provides a simple way to use pretrained models
  for various tasks, such as text classification, named entity recognition, and question answering.
- The pipeline() function takes a task name and a model name as input and returns a callable object that can be used to perform
  the task. The task name is a string that specifies the type of task to be performed, such as "sentiment-analysis" or "translation".

## BEHIND THE PIPELINE

- [1]Raw Text -> Tokenizer -> [2]Input IDs -> Model -> [3]Logits -> Post Processing -> [4]Predictions
- Ex: This course is amazing! -> [101, 102, 103] -> [0, 1, 2] -> [0.1, 0.2, 0.3] -> [positive]
- Like other neural networks, Transformer models can't process raw text directly,
  so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of.
  To do this we use a tokenizer, which will be responsible for:
  1. Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
  2. Mapping each token to an integer
  3. Adding additional inputs that may be useful to the model
