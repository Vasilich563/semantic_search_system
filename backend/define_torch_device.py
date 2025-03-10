import torch


def define_device():
    if torch.cuda.is_available():
        print("GPU is selected")
        return torch.device("cuda")
    else:
        print("CPU is selected")
        return torch.device("cpu")
