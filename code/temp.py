import torch
from discriminator import Discriminator

d = Discriminator(100, 1000)

x = torch.rand((1,1,10, 100))

y = d(x)
print("y: ", y)