import torch
from flowselect.ddlk.ddlk.data import GaussianAR1
from flowselect.modelx import ModelXKnockoff


class TestModelX:
    def test_modelx(self):
        n = 10000
        dgp = GaussianAR1(10, 0.9)
        X = dgp.sample(n=n, use_torch=True).float()
        knockoff = ModelXKnockoff(dgp.mu, dgp.Sigma)
        X_tilde = knockoff.sample(X)
        assert isinstance(X_tilde, torch.Tensor)
        assert X_tilde.size(0) == n
        assert X_tilde.size(1) == 10
        XX = torch.cat([X, X_tilde], dim=1)
        cov = XX.t().matmul(XX) / XX.size(0)
        assert torch.isclose(cov[0, 1], torch.tensor(0.9), atol=0.07)
        assert torch.isclose(cov[0, 11], torch.tensor(0.9), atol=0.07)
