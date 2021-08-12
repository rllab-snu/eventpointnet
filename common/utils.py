import torch

def CudaStatus(n = 0):
    total = torch.cuda.get_device_properties(n).total_memory / (1e9)
    reserved = torch.cuda.memory_reserved(n) / (1e9)
    allocated = torch.cuda.memory_allocated(n) / (1e9)
    free = reserved - allocated
    result = {"total": total,
              "reserved": reserved,
              "allocated": allocated,
              "free": free}
    return result