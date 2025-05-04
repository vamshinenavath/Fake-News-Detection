import torch
import torchtext
print(torch.backends.mps.is_available())  # Should print True
print(torch.backends.mps.is_built())       # Should also print True
print(torch.__version__)
