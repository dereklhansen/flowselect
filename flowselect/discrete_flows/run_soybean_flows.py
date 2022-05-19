import pandas as pd
import torch
import torch.nn.functional as F
from disc_models import *
from made import *


def get_soybeans_flows(imputed=False):
    path = "/data/gwas/soybeans/"
    start = "IMP" if imputed else "QA"
    geno_pheno = pd.read_csv(path + start + "_height.txt", sep="\t")

    genotypes = geno_pheno.iloc[:, 4:]

    genotypes = torch.as_tensor(genotypes.values, dtype=torch.int8)

    genotypes = genotypes.long()
    genotypes = F.one_hot(genotypes + 1)
    genotypes = genotypes.char()

    return genotypes


soybeans_data = get_soybeans_flows()
n, sequence_length, vocab_size = soybeans_data.shape
vector_length = sequence_length * vocab_size

num_flows = 1  # number of flow steps. This is different to the number of layers used inside each flow
temperature = (
    0.1  # used for the straight-through gradient estimator. Value taken from the paper
)
disc_layer_type = "autoreg"  #'autoreg' #'bipartite'

# This setting was previously used for the MLP and MADE networks.
nh = sequence_length + 1  # number of hidden units per layer
batch_size = 256

flows = []
for i in range(num_flows):
    if disc_layer_type == "autoreg":
        # layer = EmbeddingLayer([batch_size, sequence_length, vocab_size], output_size=vocab_size)
        # MADE network is much more powerful.
        layer = MADE([batch_size, sequence_length, vocab_size], vocab_size, [nh, nh])

        disc_layer = DiscreteAutoregressiveFlow(layer, temperature, vocab_size)

    # elif disc_layer_type == 'bipartite':
    if disc_layer_type == "bipartite":
        # MLP will learn the factorized distribution and not perform well.
        # layer = MLP(vector_length//2, vector_length//2, nh)

        layer = torch.nn.Embedding(vector_length // 2, vector_length // 2)

        disc_layer = DiscreteBipartiteFlow(
            layer, i % 2, temperature, vocab_size, vector_length, embedding=True
        )
        # i%2 flips the parity of the masking. It splits the vector in half and alternates
        # each flow between changing the first half or the second.

    flows.append(disc_layer)

model = DiscreteAutoFlowModel(flows)

print(model)

base_log_probs = torch.tensor(
    torch.randn(sequence_length, vocab_size), requires_grad=True
)
base = torch.distributions.OneHotCategorical(logits=base_log_probs)

samps = base.sample((5000,)).argmax(-1)

epochs = 1200
learning_rate = 0.001
print_loss_every = epochs // 10

losses = []
optimizer = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": learning_rate},
        {"params": base_log_probs, "lr": learning_rate},
    ]
)

model.train()
for e in range(epochs):
    batch = np.random.choice(range(4000), batch_size, False)
    x = soybeans_data[batch].float()

    if disc_layer_type == "bipartite":
        x = x.view(x.shape[0], -1)  # flattening vector

    optimizer.zero_grad()
    zs = model.forward(x)

    if disc_layer_type == "bipartite":
        zs = zs.view(
            batch_size, sequence_length, -1
        )  # adding back in sequence dimension

    base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
    # print(zs.shape, base_log_probs_sm.shape)
    logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
    loss = -torch.sum(logprob) / batch_size

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if e % print_loss_every == 0:
        print("epoch:", e, "loss:", loss.item())

x = soybeans_data[-1000:]
batch_size = x.shape[0]
zs = model.forward(x.float())
logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
loss = -torch.sum(logprob) / batch_size
print("held out loss: ", loss)
