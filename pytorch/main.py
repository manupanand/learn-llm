import torch
import numpy as np


print(torch.cuda.is_available())  # return true if  cuda available  -> true
print(torch.cuda.device_count())  # returns number of device nvidia-cuda compatiable
print(torch.cuda.get_device_name(0))  # get device  name -0 index
print(torch.cuda.device(0))  # get the device

# Tensors
# we can use numpy -numpy only run on CPU
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
# pytorch
# tensor
tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)
print(arr * 5)
print(tensor * 5)  # multiply each element
print(np.sum())  # sum of all elements
print(tensor.sum())
