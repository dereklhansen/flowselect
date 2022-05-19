import os
import sys
import torch
import torch.distributions as dist
import flowselect.flows as flows

torch.set_num_threads(2)
## Test Gaussianization flows
class TestFlows:
    def test_flows(self):
        gf_f = flows.GFKernelF(d=10, k=3)
        gf_f.mus.data.fill_(0.0)
        gf_f.log_hs.data.fill_(0.0)

        base_distribution = dist.Uniform(0, 1)
        transforms = [
            dist.SigmoidTransform().inv,
            dist.AffineTransform(loc=0.0, scale=1.0),
        ]
        logistic = dist.TransformedDistribution(base_distribution, transforms)

        Xs = torch.randn([5, 10])

        Us, logdet = gf_f.direct(Xs)
        print(Us)
        print(logistic.cdf(Xs))
        print(torch.allclose(Us, logistic.cdf(Xs)))

        Us, logdet = gf_f.direct(Xs)
        ll = logistic.log_prob(Xs).sum(1)
        print(logdet)
        print(ll)
        assert torch.allclose(logdet, ll)

        Us, logdet = gf_f.direct(Xs)
        Xs_tilde, _ = gf_f.inverse(Us)
        assert torch.allclose(Xs, Xs_tilde, atol=1e-4, rtol=1e-4)

        gf_f_unif = flows.GFKernelF(d=10, k=3, dist=dist.Uniform(0.0, 1.0))
        gf_f_unif.mus.data.fill_(0.0)
        gf_f_unif.log_hs.data.fill_(0.0)

        prob = gf_f_unif.log_prob(Xs)
        prob_logistic = logistic.log_prob(Xs).sum(dim=1)
        assert torch.allclose(prob, prob_logistic)

        gf_ig = flows.GFKernelInvG(d=10)
        Ys = torch.rand([5, 10])

        prob = gf_ig.log_prob(Ys)
        assert torch.allclose(prob, torch.tensor([0.0]))

        Us, _ = gf_ig.direct(Ys)
        Ys_tilde, _ = gf_ig.inverse(Us)
        assert torch.allclose(Ys, Ys_tilde)

        gf_rot_odd = flows.GFOrtho(7, 10)

        H = gf_rot_odd.rotation_matrix()
        assert torch.allclose(torch.det(H), torch.tensor([-1.0]))

        gf_rot_even = flows.GFOrtho(10, 10)

        H = gf_rot_even.rotation_matrix()
        assert torch.allclose(torch.det(H), torch.tensor([1.0]))

        Us, _ = gf_rot_even.direct(Ys)
        Ys_tilde, _ = gf_rot_even.inverse(Us)
        assert torch.allclose(Ys, Ys_tilde, atol=1e-4, rtol=1e-4)


# if __name__ == "__main__":
#     test_gfrot_inverse()
