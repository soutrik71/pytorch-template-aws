import torch

print(torch.cuda.is_available())  # Should return True if our GPU is enabled
print(torch.cuda.get_device_name(0))  # Should return "Tesla T4" if our GPU is enabled
print(torch.version.cuda)  # Should return "12.4" if our GPU is enabled
