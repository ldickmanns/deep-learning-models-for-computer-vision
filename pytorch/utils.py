import torch


def get_device():
    cuda_available = torch.cuda.is_available()
    cudnn_enabled = torch.backends.cudnn.enabled
    if cuda_available and cudnn_enabled:
        print('CUDA and cuDNN are available and enabled!')
        return torch.device('cuda')
    else:
        print('Running on the CPU')
        return torch.device('cpu')
