from transformers import pipeline

model_checkpoint = "6_nlp_classical_tasks/_3_translation_trainer"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
