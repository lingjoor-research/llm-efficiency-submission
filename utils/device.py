import torch


def get_device():
    """
    Returns the device to be used for training and inference.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
