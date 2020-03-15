import numpy as np
import torch
import torch.distributed as dist


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)

    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def input_preprocessing(x, device):
    x = np.transpose(x, (0, 3, 1, 2))
    x = tensor(x, device)
    x = x.float()
    x /= 255.0
    return x


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def restore_model(model, save_path):
    checkpoint = torch.load(save_path)
    model.network.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    update = checkpoint["update"]
    return update


def sync_initial_weights(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def sync_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


def cleanup():
    dist.destroy_process_group()


def sync_values(tensor_sum_values, tensor_nb_values):
    dist.reduce(tensor_sum_values, dst=0)
    dist.reduce(tensor_nb_values, dst=0)
    return tensor_sum_values / tensor_nb_values


def range_tensor(t, device):
    return torch.arange(t).long().to(device)


def zeros(shape, dtype):
    """Attempt to return torch tensor of zeros, or if numpy dtype provided,
    return numpy array or zeros."""
    try:
        return torch.zeros(shape, dtype=dtype)
    except TypeError:
        return np.zeros(shape, dtype=dtype)
