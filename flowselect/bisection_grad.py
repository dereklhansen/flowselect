import os
import sys
import pytorch_lightning as pl
import numpy as np

# import seaborn as sns
# import pandas as pd

# from matplotlib import pyplot as plt
# from tqdm.notebook import tqdm
from collections import OrderedDict

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def mul2(x):
    return x * 2


def bisection_inverse(f, u, lower, upper, max_iter=1000):
    if max_iter is None:
        max_iter = int(np.log2(upper - lower) + np.log2(1e6))
    lower = lower * torch.ones_like(u)
    upper = upper * torch.ones_like(u)
    for i in range(max_iter):
        mid = (upper + lower) / 2.0
        u_tilde = f(mid)
        tilde_lower = (u < u_tilde).float()
        lower = tilde_lower * lower + (1 - tilde_lower) * mid
        upper = tilde_lower * mid + (1 - tilde_lower) * upper
    return mid


def get_gradient(f, x):
    x = x.clone().detach().requires_grad_(True)
    u = f(x)
    u.backward(torch.ones_like(x))
    return x.grad


def bi(f, lower=1e-3, upper=1e3, max_iter=1000):
    state = {"BI_X": None, "BI_grad": None}

    class BisectionInverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            # X = bisection_inverse(lambda x: 2 * x, U, 1e-3, 1e3, 1000)
            X = bisection_inverse(f, U.clone(), lower, upper, max_iter)
            ctx.save_for_backward(X.clone())
            # BI_X
            state["BI_X"] = X.clone()
            return X

        @staticmethod
        def calc_back_grad():
            # global BI_grad
            state["BI_grad"] = get_gradient(f, state["BI_X"])

        @staticmethod
        def backward(ctx, grad_output):
            (X,) = ctx.saved_tensors
            # XX =
            # # X.detach()
            # U_again = mul2(X)
            # U_again.backward(torch.ones_like(X))
            return grad_output / state["BI_grad"]

        @staticmethod
        def bigrad():
            return state["BI_grad"]

        @staticmethod
        def bi_x():
            return state["BI_X"]

    return BisectionInverse


if __name__ == "__main__":
    f = lambda x: 2 * x
    BI = bi(f)
    fi = BI.apply
    x = torch.randn(5)
    x.requires_grad = True
    y = fi(x)
    print(y)
    BI.calc_back_grad()
    y.backward(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    g = x.grad
    print(g)
