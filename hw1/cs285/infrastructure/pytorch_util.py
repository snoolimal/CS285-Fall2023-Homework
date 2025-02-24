import numpy
import torch

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
        print('Using GPU id {}'.format(gpu_id))
    else:
        device = torch.device('cpu')
        print('GPU not detected. Defaulting to CPU.')


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def to_tensor(array: numpy.ndarray):
    return torch.from_numpy(array).float().to(device)


def to_numpy(tensor: torch.Tensor):
    return tensor.to('cpu').detach().numpy()