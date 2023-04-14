import torch

# create the tensors
tensor1 = torch.randn(1, 1, 2)
tensor2 = torch.randn(1, 2, 2)
print(tensor1)

# expand tensor1 along the second dimension
tensor1 = tensor1.expand(1, 2, 2)

result = tensor1 + tensor2
print(tensor1)

print(tensor2)

print(result)
print(result.shape)