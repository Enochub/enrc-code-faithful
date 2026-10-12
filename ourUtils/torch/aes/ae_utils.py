import torch
import torch.nn.functional as F
import numpy as np


def encode_batchwise(model, data_loader, device=torch.device('cpu')):
    encodings = []
    for batch in data_loader:
        batch_data = batch[0].to(device)
        z = model.encode(batch_data)
        encodings.append(z.detach().cpu())
        torch.cuda.empty_cache()
    encodings = torch.cat(encodings, 0)
    return encodings


def decode_batchwise(model, data_loader, device=torch.device('cpu')):
    decodings = []
    for batch in data_loader:
        batch_data = batch[0].to(device)
        z = model.encode(batch_data)
        rec = model.decode(z)
        decodings.append(rec.detach().cpu())
        torch.cuda.empty_cache()
    decodings = torch.cat(decodings, 0)
    return decodings


def encode_decode_batchwise(model, data_loader, device=torch.device('cpu')):
    encodings = []
    decodings = []
    for batch in data_loader:
        batch_data = batch[0].to(device)
        z = model.encode(batch_data)
        rec = model.decode(z)
        encodings.append(z.detach().cpu())
        decodings.append(rec.detach().cpu())
        torch.cuda.empty_cache()
    encodings = torch.cat(encodings, 0)
    decodings = torch.cat(decodings, 0)

    return encodings, decodings


def eval_batchwise(model, data_loader, loss_fn, device=torch.device('cpu')):
    model.eval()
    rec_loss = 0
    i = 0
    for batch in data_loader:
        batch_data = batch[0].to(device)
        z = model.encode(batch_data)
        rec = model.decode(z)
        rec_loss += loss_fn(rec, batch_data).item()
        i += 1
        torch.cuda.empty_cache()
    return rec_loss / i
