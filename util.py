import torch.cuda as cuda
import torch.backends.mps as mps


def get_available_device():
    if cuda.is_available():
        return 'cuda'
    elif mps.is_available():
        return 'mps'
    else:
        return 'cpu'