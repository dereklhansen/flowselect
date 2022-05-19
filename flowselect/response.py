import numpy as np


def identity(x):
    return x


class Response:
    def __init__(self, signal_n, signal_a, covariate_fs=None):
        self.signal_n = signal_n
        self.signal_a = signal_a

        if covariate_fs is None:
            covariate_fs = [identity] * signal_n
        else:
            assert len(covariate_fs) == signal_n
        self.covariate_fs = covariate_fs

    def __call__(self, X, beta_nonzero_in=[]):
        X = X.copy()
        n, p = X.shape
        beta = np.zeros((p, 1))
        fs = np.array([identity] * p)
        betas_remaining = [j for j in range(p) if not (j in beta_nonzero_in)]
        beta_nonzero_sampled = np.random.choice(
            betas_remaining, self.signal_n - len(beta_nonzero_in), replace=False
        )
        if len(beta_nonzero_in) > 0:
            beta_nonzero = np.concatenate((beta_nonzero_in, beta_nonzero_sampled))
        else:
            beta_nonzero = beta_nonzero_sampled
        print(beta_nonzero)
        beta[beta_nonzero, 0] = (
            (2 * np.random.choice(2, self.signal_n) - 1) * self.signal_a / np.sqrt(n)
        )
        fs[beta_nonzero] = self.covariate_fs
        for idx in beta_nonzero:
            X[:, idx] = fs[idx](X[:, idx])
        y = np.dot(X, beta) + np.random.normal(size=(n, 1))
        return X, y.flatten(), {"beta": beta, "fs": fs}


def sin5(x):
    return np.sin(5.0 * x)


def cos5(x):
    return np.cos(5.0 * x)


def make_response(conf):
    signal_n = conf.signal_n
    signal_a = conf.signal_a
    in_str = conf.name
    if in_str == "linear":
        covariate_fs = None
    elif in_str == "sin_cos":
        sig_n_half = signal_n // 2
        covariate_fs = [np.sin] * sig_n_half + [np.cos] * (signal_n - sig_n_half)
    elif in_str == "sin_cos5":
        sig_n_half = signal_n // 2
        covariate_fs = [sin5] * sig_n_half + [cos5] * (signal_n - sig_n_half)
    else:
        raise (Exception("Response type not recognized"))
    return Response(signal_n=signal_n, signal_a=signal_a, covariate_fs=covariate_fs)
