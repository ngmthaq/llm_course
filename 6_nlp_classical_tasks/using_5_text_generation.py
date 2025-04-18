import torch
from transformers import pipeline

# Get the device to use for inference
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the text generation pipeline
pipe = pipeline(
    "text-generation", model="huggingface-course/codeparrot-ds", device=device
)

# Example text to generate code from
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""

# Generate code
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
