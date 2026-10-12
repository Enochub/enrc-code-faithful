import numpy as np
import torch
import random


def random_seed(seed_value, use_cuda=True):
    seed_value = int(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def finetune(learn, n_iterations, lr):
    # Deactivate Batchnorm and dropout
    learn.model.eval()
    optimizer = learn.opt_func(learn.model.parameters(), lr=lr)
    i = 0
    while(i < n_iterations):
        for batch in learn.data.train_dl:
            device = next(learn.model.parameters()).device
            batch = batch[0].to(device)
            z = learn.model.encode(batch)
            reconstructed = learn.model.decode(z)
            rec_loss = torch.nn.functional.mse_loss(reconstructed, batch)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
            i += 1
            if i % 100 == 0:
                print(f"Iter {i}: loss: {rec_loss.item():.4f}")
            if i > n_iterations:
                break
