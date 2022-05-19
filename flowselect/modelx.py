import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from rpy2.robjects.packages import importr
from tqdm import tqdm
## Automatically convert numpy arrays to R objects
## when calling R methods

from rpy2.robjects import numpy2ri

numpy2ri.activate()


class ModelXKnockoff:
    def __init__(self, mu, sigma):
        self.knockoff = importr("knockoff")
        self.mu = mu
        self.sigma = sigma

    def sample(self, X):
        X_tilde = self.knockoff.create_gaussian(X.numpy(), self.mu, self.sigma)
        X_tilde_np = np.array(X_tilde)
        return torch.from_numpy(X_tilde_np).float()

    def to(self, device):
        assert device.type == "cpu"
        return self


class ModelXKnockoffMixture:
    def __init__(self, mus, sigmas, pis, deterministic=True):
        self.deterministic = deterministic
        self.pis_torch = torch.from_numpy(pis)

        self.mixture_dists = []
        self.knockoff_dists = []
        for mu, sigma in zip(mus, sigmas):
            self.mixture_dists.append(
                MultivariateNormal(torch.from_numpy(mu), torch.from_numpy(sigma))
            )
            self.knockoff_dists.append(ModelXKnockoff(mu, sigma))

    def assignment_probs(self, X):
        log_probs = []
        for pi, dist in zip(self.pis_torch, self.mixture_dists):
            log_probs.append(dist.log_prob(X) + pi.log())
        log_probs = torch.stack(log_probs, dim=1)
        assignment = log_probs.softmax(dim=1)
        return assignment

    def sample(self, X):
        assignment_prob = self.assignment_probs(X)
        if self.deterministic:
            idxs = torch.argmax(assignment_prob, dim=1)
        else:
            print("INFO: Sampling mixture assignments...")
            cat_dist = Categorical(assignment_prob)
            idxs = cat_dist.sample()
        X_tilde = X.clone()
        for i, knockoff_dist in tqdm(enumerate(self.knockoff_dists), desc="Evaluating knockoffs per component"):
            obs_to_select = (idxs == i)
            if obs_to_select.sum() > 0:
                Xi = X[obs_to_select]
                Xi_tilde = knockoff_dist.sample(Xi)
                X_tilde[obs_to_select] = Xi_tilde
        return X_tilde

    def to(self, device):
        assert device.type == "cpu"
        return self
