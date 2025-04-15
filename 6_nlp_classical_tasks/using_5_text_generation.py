import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipe = pipeline(
    "text-generation", model="huggingface-course/codeparrot-ds", device=device
)

txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""

print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
