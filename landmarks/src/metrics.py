import torch

def nme(y_pred, y_true, ds):
    sqerr = torch.sqrt((((y_true - y_pred)**2)).sum(-1))
    if ds is not None:
        sqerr = sqerr * ds
    return sqerr.mean()