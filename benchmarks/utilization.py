import torch
import psutil


def gpu_utilization():
    if torch.cuda.is_available():
        return torch.cuda.utilization()
    return None


def cpu_utilization(interval=0.05):
    return psutil.cpu_percent(interval=interval)