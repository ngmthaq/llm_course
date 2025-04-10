## UNDERSTANDING NLP AND LLMs:

- NLP (Natural Language Processing) is the broader field focused on enabling computers to understand, interpret,
  and generate human language. NLP encompasses many techniques and tasks such as sentiment analysis, named entity recognition,
  and machine translation.
- LLMs (Large Language Models) are a powerful subset of NLP models characterized by their massive size,
  extensive training data, and ability to perform a wide range of language tasks with minimal task-specific training.
  Models like the Llama, GPT, or Claude series are examples of LLMs that have revolutionized what's possible in NLP.

## TRANSFORMER MODELS:

- Transformer models are a type of neural network architecture that has become the foundation for many state-of-the-art NLP models.
- Transformer models are used to solve all kinds of NLP tasks, like the ones mentioned in the previous section.
  Here are some of the companies and organizations using Hugging Face and Transformer models,
  who also contribute back to the community by sharing their models.
- The 🤗 Transformers library provides the functionality to create and use those shared models.
  The Model Hub contains thousands of pretrained models that anyone can download and use.
  You can also upload your own models to the Hub!
- The first tool from the 🤗 Transformers library: the pipeline() function.
- Some of the currently available pipelines are:
  - feature-extraction (get the vector representation of a text)
  - fill-mask
  - ner (named entity recognition)
  - question-answering
  - sentiment-analysis
  - summarization
  - text-generation
  - translation
  - zero-shot-classification

## PRE-TRAINING

- Pre-training is the act of training a model from scratch: the weights are randomly initialized,
  and the training starts without any prior knowledge.
- The output of the pre-training phase is a model that has learned to understand the language it was trained on.
- The model is not yet able to perform any specific tasks, but it has a general understanding of the language.
- This model name now is called a "general pretrained model" or "pretrained model".

## SELF-SUPERVISED LEARNING

- Self-supervised learning is a type of machine learning where the model learns from unlabeled data by creating its own labels.
- In the context of NLP, self-supervised learning is often used to pre-train models on large corpora of text data.
- This type of model develops a statistical understanding of the language it has been trained on,
  but it's not very useful for specific practical tasks.
- Because of this, the general pretrained model then goes through a process called transfer learning.

## TRANSFER LEARNING

- Because of the self-supervised learning process, the model has a general understanding of the language it was trained on,
  but it doesn't know how to perform specific tasks. The general pretrained model is going to the next step: transfer learning.
- Transfer learning is the process of taking a pretrained model and fine-tuning it on a specific task.

## FINE-TUNING

- Fine-tuning is the process of taking a pretrained model and training it on a specific task with labeled data.
- The model is trained on a smaller dataset that is specific to the task at hand.
- The output of the fine-tuning phase is a model that is able to perform the specific task it was trained on.
- The model is now called a "fine-tuned model" or "task-specific model".
- The fine-tuned model is now able to perform the specific task it was trained on, but it still has a general understanding of the language.

## GENERAL ARCHITECTURE

- The general architecture of a transformer model consists of an encoder and a decoder.
- The encoder takes the input text and converts it into a vector representation.
- The decoder takes the vector representation and converts it back into text.

## ENCODER MODELS

- Encoder models are transformer models that only use the encoder part of the architecture.
- Encoder models are used for tasks like text classification, where the model needs to understand the input text
  and make a prediction based on it.

## DECODER MODELS

- Decoder models are transformer models that only use the decoder part of the architecture.
- Decoder models are used for tasks like text generation, where the model needs to generate text based on a given input.

## SEQUENCE TO SEQUENCE MODELS

- Sequence to sequence models are transformer models that use both the encoder and decoder parts of the architecture.
- Sequence to sequence models are used for tasks like machine translation, where the model needs to take an input sequence
  and generate an output sequence.

## ARCHITECTURE VS. CHECKPOINTS

- As we dive into Transformer models in this course, you'll see mentions of architectures and checkpoints as well as models.
  These terms all have slightly different meanings:
  - Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
  - Checkpoints: These are the weights that will be loaded in a given architecture.
  - Model: This is an umbrella term that isn't as precise as “architecture” or “checkpoint”: it can mean both. This course will specify architecture or checkpoint when it matters to reduce ambiguity.
  - For example, BERT is an architecture while bert-base-cased, a set of weights trained by the Google team for the first release of BERT, is a checkpoint.
    However, one can say “the BERT model” and “the bert-base-cased model.”
