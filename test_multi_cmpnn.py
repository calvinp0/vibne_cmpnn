
# -----------------------------------------------------------------------------#
#  PyTest **sanity-suite** for the paired-molecule CMPNN implementation
# -----------------------------------------------------------------------------#
#  What we are checking (ordered roughly from cheap → expensive):
#
#  1. **Dataset integrity**
#     • every CSV row (= a pair of SMILES) yields exactly *one* item in the
#       `MultiCMPNNDataset`; each item contains **two** PyG `Data` objects.  
#     • the target `y` is duplicated correctly so that both graphs in a pair
#       carry the identical label.
#
#  2. **Custom collate logic**
#     • `collate_pairs()` must return a *single* `torch_geometric.data.Batch`
#       – *not* a list/tuple – whose `num_graphs` equals `2 × n_pairs`.
#
#  3. **Encoder plumbing**
#     • shared-encoder (`mpn_shared=True`) and twin-encoder
#       (`mpn_shared=False`) paths both produce an embedding matrix with the
#       expected shape:  
#         `(num_graphs , embed_dim)` where  
#         `embed_dim = 2·hidden_dim` if GRU read-out is used, else `hidden_dim`.
#     • when twin encoders are requested their parameters must *not* be
#       identical, confirming two independent networks are instantiated.
#
#  4. **Forward pass semantics**
#     • calling the LightningModule with a *Batch* that contains `2 × N` graphs
#       must return `N` predictions – one per molecular pair.  
#     • (xfail) A legacy swap-invariance test is kept but marked as *expected to
#       fail* because the model is **order-sensitive by design** (r1H vs r2H).
#
#  5. **Tiny over-fit smoke test**
#     • running a handful of gradient steps on a *single* pair must reduce the
#       MSE loss by ≥ 80 %; this asserts that gradients flow end-to-end and the
#       network can in principle learn.
#
#  6. ───────── Public-API guardrails
#     • verifies that `collate_pairs` still returns a `Batch` *only* and that
#       the module’s public `forward` works with either shared or twin encoders.
#     • explicit “order-sensitivity” test: swapping the two graphs in a pair
#       should change the prediction by a noticeable margin (≥ 1 e-4).
#
#  Together these tests give confidence that:
#       – data ► model wiring is correct,
#       – both encoder configurations behave as expected,
#       – the model honours molecule order where that matters, and
#       – the overall training loop can learn on at least a toy example.
# -----------------------------------------------------------------------------#



import math
import torch
import pytest
from torch_geometric.data import Batch

from dataset       import MultiCMPNNDataset
from pl_module     import MultiCMPNNLitModel
from utils_paired  import collate_pairs


@pytest.fixture(scope="module")
def full_ds():
    csv = "raw/multi_mols.csv"         # <-- adjust if file lives elsewhere
    return MultiCMPNNDataset(root=".", csv_file="multi_mols.csv",
                             atom_messages=True, force_reload=True)


@pytest.fixture(scope="module")
def tiny_batch(full_ds):
    # first two pairs  →  4 graphs
    return collate_pairs([full_ds[i] for i in range(2)])


# ------------------------------------------------------------------------- #
# 1 ───────── Dataset integrity                                             #
# ------------------------------------------------------------------------- #
def test_pairs_per_row(full_ds):
    import pandas as pd
    df = pd.read_csv("raw/multi_mols.csv")
    assert len(full_ds) == len(df)              # 1 CSV row ↔ 1 pair (2 graphs)

def test_y_copied(full_ds):
    g1, g2 = full_ds[0]
    assert torch.allclose(g1.y, g2.y)


# ------------------------------------------------------------------------- #
# 2 ───────── Custom collate returns a Batch                                #
# ------------------------------------------------------------------------- #
def test_collate_returns_batch(tiny_batch):
    assert isinstance(tiny_batch, Batch)
    assert tiny_batch.num_graphs == 4
    assert tiny_batch.x.size(0) == tiny_batch.batch.size(0)


# ------------------------------------------------------------------------- #
# 3 ───────── Encoder paths                                                 #
# ------------------------------------------------------------------------- #
@pytest.mark.parametrize("shared", [True, False])
def test_encoder_output_shape(full_ds, tiny_batch, shared):
    n_atom = full_ds.num_atom_features
    n_bond = full_ds.num_bond_features

    model = MultiCMPNNLitModel(
        in_node_feats=n_atom,
        in_edge_feats=n_bond,
        hidden_dim=64,
        num_steps=2,
        mpn_shared=shared,
    )

    z = model._encode_big_batch(tiny_batch)
    assert z.shape[0] == tiny_batch.num_graphs            # 2N rows
    # embedding size is 2*hidden if GRU read-out
    expect_dim = 2 * 64 if model.hparams.readout == "gru" else 64
    assert z.shape[1] == expect_dim

def test_twin_encoders_different(full_ds):
    n_atom = full_ds.num_atom_features
    n_bond = full_ds.num_bond_features

    m = MultiCMPNNLitModel(in_node_feats=n_atom, in_edge_feats=n_bond,
                           mpn_shared=False, hidden_dim=32, num_steps=1)
    # identical shapes but parameters should not all be equal
    diff = sum((p1 != p2).sum().item()
               for p1, p2 in zip(m.encoder1.parameters(),
                                 m.encoder2.parameters()))
    assert diff > 0


# ------------------------------------------------------------------------- #
# 4 ───────── Forward pass & swap-invariance (shared encoder)               #
# ------------------------------------------------------------------------- #
def test_forward_shapes(full_ds, tiny_batch):
    n_atom = full_ds.num_atom_features
    n_bond = full_ds.num_bond_features
    model  = MultiCMPNNLitModel(in_node_feats=n_atom, in_edge_feats=n_bond,
                                hidden_dim=64, num_steps=2, mpn_shared=True)
    y_hat = model(tiny_batch)         # forward expects a Batch only
    assert y_hat.ndim == 1
    assert y_hat.numel()*2 == tiny_batch.num_graphs

@pytest.mark.xfail(reason="Model is order-sensitive by design")
def test_swap_invariance(full_ds):
    n_atom = full_ds.num_atom_features
    n_bond = full_ds.num_bond_features
    model  = MultiCMPNNLitModel(in_node_feats=n_atom, in_edge_feats=n_bond,
                                hidden_dim=64, num_steps=2, mpn_shared=True)

    # build a two-pair batch and its molecule-order swap
    pairs = [full_ds[i] for i in range(2)]
    batch_A = collate_pairs(pairs)
    batch_B = collate_pairs([(p[1], p[0]) for p in pairs])   # swap mols

    with torch.no_grad():
        delta = (model(batch_A) - model(batch_B)).abs().max().item()

    # allow tiny numerical noise only
    assert delta < 1e-5


# ------------------------------------------------------------------------- #
# 5 ───────── Over-fit smoke test (tiny optimisation)                       #
# ------------------------------------------------------------------------- #
def test_overfit_one_pair(full_ds):
    # one training step on a single pair should drive loss ↓↓↓
    n_atom = full_ds.num_atom_features
    n_bond = full_ds.num_bond_features
    model  = MultiCMPNNLitModel(in_node_feats=n_atom, in_edge_feats=n_bond,
                                hidden_dim=32, num_steps=1, mpn_shared=True)
    pair_batch = collate_pairs([full_ds[0]])

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit  = torch.nn.MSELoss()

    before = crit(model(pair_batch), pair_batch.y.view(-1)).item()
    for _ in range(50):
        optim.zero_grad()
        loss = crit(model(pair_batch), pair_batch.y.view(-1))
        loss.backward(); optim.step()
    after  = crit(model(pair_batch), pair_batch.y.view(-1)).item()

    assert after < before * 0.2          # should fit ~80 % better

# test_api_latest.py
import torch
import pytest
from torch_geometric.data import Data, Batch
from utils_paired import collate_pairs
from pl_module     import MultiCMPNNLitModel


# -------------------------------------------------------------------- helpers
def make_graph(n_node_feat: int, n_edge_feat: int) -> Data:
    x          = torch.randn(2, n_node_feat)
    edge_index = torch.tensor([[0, 1, 1, 0],
                               [1, 0, 0, 1]], dtype=torch.long)
    edge_attr  = torch.randn(edge_index.size(1), n_edge_feat)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([0.5]))


# -------------------------------------------------------------------- tests
def test_collate_pairs_batch_only():
    g1 = make_graph(4, 5)
    g2 = make_graph(4, 5)
    big_batch = collate_pairs([(g1, g2)])     # NEW: returns Batch only
    assert isinstance(big_batch, Batch)
    assert big_batch.num_graphs == 2


@pytest.mark.parametrize("shared", [True, False])
def test_forward_two_pairs(shared):
    in_nf, in_ef = 3, 4
    pairs  = [(make_graph(in_nf, in_ef), make_graph(in_nf, in_ef))
              for _ in range(2)]              # 2 pairs → 4 graphs
    batch  = collate_pairs(pairs)

    model = MultiCMPNNLitModel(
        in_node_feats=in_nf,
        in_edge_feats=in_ef,
        hidden_dim=16, num_steps=1,
        mpn_shared=shared,
    )
    out = model(batch)                        # Batch only
    assert out.ndim == 1 and out.numel() == 2



def make_graph(n_atom, n_bond):
    x = torch.randn(2, n_atom)
    edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]])
    edge_attr  = torch.randn(edge_index.size(1), n_bond)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([0.3]))


@pytest.mark.parametrize("mpn_shared", [True, False])
def test_order_sensitivity(mpn_shared):
    n_atom, n_bond = 5, 6
    g1, g2 = make_graph(n_atom, n_bond), make_graph(n_atom, n_bond)

    batch_AB = collate_pairs([(g1, g2)])         # (r1h=rA, r2h=rB)
    batch_BA = collate_pairs([(g2, g1)])         # swapped

    model = MultiCMPNNLitModel(
        in_node_feats=n_atom,
        in_edge_feats=n_bond,
        hidden_dim=32,
        num_steps=1,
        mpn_shared=mpn_shared,
    ).eval()

    with torch.no_grad():
        out_AB = model(batch_AB)    # tensor([x])
        out_BA = model(batch_BA)    # tensor([y])

    # They should *not* be (almost) equal
    assert (out_AB - out_BA).abs().item() > 1e-4


def test_order_sensitivity(full_ds):
    n_atom = full_ds.num_atom_features
    n_bond = full_ds.num_bond_features
    model = MultiCMPNNLitModel(in_node_feats=n_atom, in_edge_feats=n_bond, hidden_dim=64, num_steps=2, mpn_shared=True).eval()
    pairs = [full_ds[i] for i in range(2)]
    batch_A = collate_pairs(pairs) # (r1h, r2h)
    batch_B = collate_pairs([(p[1], p[0]) for p in pairs]) # 
    with torch.no_grad():
        delta = (model(batch_A) - model(batch_B)).abs().max().item()
    assert delta > 1e-3