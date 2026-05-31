import torch


print(torch.cuda.is_available())  # return true if  cuda available  -> true
print(torch.cuda.device_count())  # returns number of device nvidia-cuda compatiable
print(torch.cuda.get_device_name(0))  # get device  name -0 index
