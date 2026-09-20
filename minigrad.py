import torch

a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([2.0], requires_grad=True)
c = a + b
d = c * c
loss = d + d

loss.backward(retain_graph=True)                            # Retain the computational graph
print(a.grad, b.grad, c, d, loss)
