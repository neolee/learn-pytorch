import torch
import torch.cuda as cuda
import torch.backends.mps as mps


def show_backend_info():
    print('PyTorch version:', torch.__version__)

    print('CUDA available:', cuda.is_available())
    print('MPS available:', mps.is_available() and mps.is_built())


def get_available_device():
    if cuda.is_available():
        return 'cuda'
    elif mps.is_available():
        return 'mps'
    else:
        return 'cpu'