import torch


def get_numpy(x):
    return x.detach().cpu().numpy()

def int_to_one_hot(label_tensor, n_labels):
    """
    Creates a one_hot_matrix for the given label vector.
    """
    onehot = torch.zeros([label_tensor.shape[0], n_labels], device=label_tensor.device)
    onehot.scatter_(1, label_tensor.unsqueeze(1), 1.0)
    return onehot
