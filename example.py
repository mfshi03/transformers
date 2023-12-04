import torch

output = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

max_values = output[0].max(dim=1)
print(max_values)