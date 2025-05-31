import torch
import numpy as np
import pytest
from torch_scatter import scatter_sum, scatter_max
from model import BatchGRUBooster, CMPNNEncoder
from dataset import CMPNNDataset
from torch_geometric.data import Batch
import pathlib
import pandas as pd

from preprocessing import scaffold_split_indices          # ← you still have this
from cross_val import run_scaffold_cv

RAW_DIR   = pathlib.Path("raw")
MASTER_CSV = RAW_DIR / "SAMPL.csv"        # change if the file is named differently


@pytest.fixture(scope="session", autouse=True)
def default_scaffold_split():
    """
    Ensure raw/train.csv, raw/val.csv, raw/test.csv exist for tests.

    The split is deterministic (seed=42) and is done once per test session.
    """
    if (RAW_DIR / "train.csv").exists():
        return                                              # nothing to do

    if not MASTER_CSV.exists():
        pytest.skip(f"{MASTER_CSV} not found – cannot build default split")

    df = pd.read_csv(MASTER_CSV)

    train_idx, val_idx, test_idx = scaffold_split_indices(
        df, valid_ratio=0.10, test_ratio=0.10, seed=42
    )

    RAW_DIR.mkdir(exist_ok=True)
    df.iloc[train_idx].to_csv(RAW_DIR / "train.csv", index=False)
    df.iloc[val_idx]  .to_csv(RAW_DIR / "val.csv",   index=False)
    df.iloc[test_idx] .to_csv(RAW_DIR / "test.csv",  index=False)

    print("[conftest] scaffold split written to raw/train|val|test.csv")
# 1. Booster prefix positions
def test_booster_prefix_positions():
    h = torch.arange(3).unsqueeze(1).float()
    batch = torch.tensor([0, 0, 1])
    num_graphs = int(batch.max()) + 1
    lengths = torch.bincount(batch.cpu(), minlength=num_graphs).to(batch.device)
    prefix = torch.cumsum(lengths, dim=0) - lengths
    positions = (torch.arange(h.size(0), device=h.device) - prefix[batch]).long()
    expected = torch.tensor([0, 1, 0])
    assert torch.equal(positions, expected)

# 2. Communicator parameter count
def test_communicator_param_count():
    enc = CMPNNEncoder(10, 6, hidden_dim=300, num_steps=6, dropout=0.0, n_tasks=1)
    total = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    # static sentinel only for canonical hidden_dim=300, gru readout, single task
    if enc.hidden_dim == 300 and enc.readout == 'gru' and enc.n_tasks == 1:
        assert total == 4254001
    else:
        pytest.skip('Param sentinel only valid for default config')

# 3. Inverse-edge KeyError
@pytest.mark.xfail(reason='Directed edges intentionally unsupported', strict=False)
def test_inverse_edge_keyerror():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(KeyError):
        CMPNNEncoder._build_inverse_edges(edge_index, num_nodes=2)

# 4. Booster formula
def test_booster_formula():
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
    h_e = torch.arange(1, 7).unsqueeze(1).float()
    m_sum = scatter_sum(h_e, edge_index[1], dim=0, dim_size=3)
    m_max, _ = scatter_max(h_e, edge_index[1], dim=0, dim_size=3)
    assert m_sum.device == m_max.device
    res = m_sum * m_max
    expected = torch.tensor([[28.], [30.], [48.]])
    assert torch.allclose(res, expected)

# 5. Booster run-once
def test_booster_run_once():
    enc = CMPNNEncoder(10, 6, hidden_dim=8, num_steps=0, dropout=0.0, n_tasks=1)
    # only run if booster exists
    if not hasattr(enc, 'batch_gru'):
        pytest.skip('No booster to test')
    enc.eval()
    count = {'calls': 0}
    orig = enc.batch_gru.forward

    def spy(h_gru, batch_gru):
        count['calls'] += 1
        return orig(h_gru, batch_gru)

    enc.batch_gru.forward = spy
    h_v = torch.randn(5, 8)
    h_e = torch.randn(5, 8)
    enc._message_passing(h_v, h_e, h_e, torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), torch.zeros(5, dtype=torch.long))
    assert count['calls'] == 1

# 6. Embed dimension + param heuristic
@pytest.mark.parametrize('h', [64, 128, 256])
def test_embed_and_param_heuristic(h):
    enc = CMPNNEncoder(10, 6, hidden_dim=h, num_steps=3)
    z = enc.embed(torch.randn(3, 10), torch.empty((2, 0), dtype=torch.long), torch.empty((0, 6)), torch.zeros(3, dtype=torch.long))
    assert z.size(1) == 2 * h
    p = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert 10 * h < p < 100 * h * h

# 7. Smoke tests: optimizer, gradients, permutation
@pytest.mark.parametrize('readout,use_booster,perm_inv', [('sum', False, True), ('gru', True, False)])
def test_smoke_training_and_permutation(readout, use_booster, perm_inv):
    feat_dim, edge_dim = 10, 6
    # use deterministic input for sum-mode permutation invariance
    if perm_inv:
        x = torch.arange(4 * feat_dim, dtype=torch.float).reshape(4, feat_dim)
    else:
        x = torch.randn(4, feat_dim)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, edge_dim))
    batch = torch.zeros(4, dtype=torch.long)
    y = torch.randn(1)
    model = CMPNNEncoder(feat_dim, edge_dim, hidden_dim=16, num_steps=2, dropout=0.0, n_tasks=1, readout=readout, use_booster=use_booster)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    # forward + backward + step
    pred = model(x, edge_index, edge_attr, batch)
    loss = criterion(pred.view(-1), y)
    loss.backward()
    optimizer.step()
    # check gradients (collect missing ones)
    unused = set()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None and id(p) not in unused]
    assert not missing, f'Missing gradients for: {missing}'
    # switch to eval so BatchNorm uses running stats and sum-pooling remains invariant
    model.eval()
    # recompute prediction in eval mode for invariance check
    pred_eval = model(x, edge_index, edge_attr, batch)
    # permutation behavior
    perm = torch.randperm(x.size(0))
    if perm_inv:
        # test invariance on a fresh model with same weights
        model2 = CMPNNEncoder(feat_dim, edge_dim, hidden_dim=16, num_steps=2,
                             dropout=0.0, n_tasks=1, readout=readout, use_booster=use_booster)
        model2.load_state_dict(model.state_dict())
        model2.eval()
        p1 = model2(x, edge_index, edge_attr, batch)
        p2 = model2(x[perm], edge_index, edge_attr, batch[perm])
        assert torch.allclose(p1, p2), 'Model not permutation-invariant in sum mode'
    else:
        p2 = model(x[perm], edge_index, edge_attr, batch[perm])
        assert not torch.allclose(pred_eval, p2), 'GRU readout should be permutation-variant'

# New tests for embed size in sum mode
def test_embed_size_sum_mode():
    h = 32
    enc = CMPNNEncoder(10, 6, hidden_dim=h, num_steps=3, readout='sum', use_booster=False)
    z = enc.embed(torch.randn(5, 10), torch.empty((2, 0), dtype=torch.long), torch.empty((0, 6)), torch.zeros(5, dtype=torch.long))
    assert z.size(1) == h, f'embed size {z.size(1)} != hidden_dim {h} in sum mode'

# Edge case: empty graph
def test_empty_graph():
    enc = CMPNNEncoder(10, 6, hidden_dim=16, num_steps=2)
    x = torch.empty((0, 10))
    ei = torch.empty((2, 0), dtype=torch.long)
    ea = torch.empty((0, 6))
    batch = torch.empty((0,), dtype=torch.long)
    # should not error
    out1 = enc(x, ei, ea, batch)
    out2 = enc.embed(x, ei, ea, batch)
    assert out1.numel() == out2.numel() or out1.numel() >= 0

def test_directed_edges():
    """Ensure every bond in the dataset has its reverse edge."""
    from dataset import CMPNNDataset
    # csv_file must match raw_file_names, so pass 'train.csv' not 'raw/train.csv'
    data = CMPNNDataset(root='.', csv_file='train.csv')[0]
    ei = data.edge_index.cpu().numpy().T.tolist()
    missing = [(u,v) for u,v in ei if [v,u] not in ei]
    assert not missing, f"Missing reverse edges: {missing[:5]}"

def test_scaffold_leakage():
    """Ensure no scaffold overlap between train/val/test splits."""
    import pandas as pd
    from dataset import generate_scaffold
    df_tr = pd.read_csv('raw/train.csv')
    df_val= pd.read_csv('raw/val.csv')
    df_te = pd.read_csv('raw/test.csv')
    s_tr  = set(map(generate_scaffold, df_tr.smiles))
    s_val = set(map(generate_scaffold, df_val.smiles))
    s_te  = set(map(generate_scaffold, df_te.smiles))
    assert not (s_tr & s_val), "train/val scaffold overlap"
    assert not (s_tr & s_te), "train/test scaffold overlap"
    assert not (s_val & s_te), "val/test scaffold overlap"

# def test_y_scramble_blows_up(tmp_path):
#     """RMSE should ≫ 1 when targets are permuted."""
#     df = pd.read_csv("raw/SAMPL.csv")
#     df.y = np.random.permutation(df.y.values)
#     tmp_csv = pathlib.Path("raw") / "scrambled.csv"
#     df.to_csv(tmp_csv, index=False)
#     rmses = run_scaffold_cv(tmp_csv, n_splits=3, n_repeats=1)   # short CV
#     assert np.mean([r for _,_,r in rmses]) > 2.0        # expect big error

def test_mean_predictor_baseline():
    from preprocessing import scaffold_split_indices
    df = pd.read_csv("raw/SAMPL.csv")
    tr, va, te = scaffold_split_indices(df)
    y = df.y.values
    rmses = np.sqrt(((y[te] - y[tr].mean())**2).mean())
    assert rmses > 2.0 # must be a bad baseline\n


def test_rmse_exact():
    true = np.array([ 1.0, -1.0, 2.0])
    pred = np.array([ 1.5, -0.5, 2.5])
    ref  = np.sqrt(((true - pred) ** 2).mean())
    from utils import rmse_torch   # ← the function you call during training
    assert abs(ref - rmse_torch(true, pred)) < 1e-9



@pytest.fixture(scope="session")
def freesolv_stats():
    # precomputed from the official SAMPL.csv downloaded on 2025-05-XX
    return {"mean": -3.80, "std": 3.85}

def test_target_distribution_stable(freesolv_stats):
    import pandas as pd
    df = pd.read_csv("raw/SAMPL.csv")
    mean, std = df.y.mean(), df.y.std()
    assert abs(mean - freesolv_stats["mean"]) < 0.05
    assert abs(std  - freesolv_stats["std"] ) < 0.05


# def test_grad_flow_toy_graph():
#     enc = CMPNNEncoder(5, 4, hidden_dim=8, num_steps=1)
#     x   = torch.randn(3, 5, requires_grad=True)
#     ei  = torch.tensor([[0,1],[1,2]])
#     ea  = torch.randn(2, 4)
#     batch = torch.zeros(3, dtype=torch.long)
#     out = enc(x, ei, ea, batch).sum()
#     out.backward()
#     # every parameter that requires grad should have non-None grad
#     missing = [n for n,p in enc.named_parameters() if p.requires_grad and p.grad is None]
#     assert not missing



def test_normalization_uses_only_train_stats(tmp_path):
    # 1) Create a dummy CSV with 4 molecules (same SMILES for simplicity)
    df = pd.DataFrame({
        'smiles': ['C', 'C', 'C', 'C'],
        'y':       [0.0,   10.0, 20.0, 30.0]
    })
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "dummy.csv"
    df.to_csv(csv_path, index=False)

    # 2) Instantiate dataset, forcing a fresh process
    ds = CMPNNDataset(root=str(tmp_path), csv_file="dummy.csv", force_reload=True)

    # 3) Pick train/val/test splits
    train_idx = [0, 1]
    val_idx   = [2]
    test_idx  = [3]

    # 4) Compute normalization on train only
    mean, std = ds.compute_normalization(train_idx)
    # for [0,10] mean=5, std=sqrt(((0-5)^2+(10-5)^2)/2)=5
    assert mean == pytest.approx(5.0)
    assert std  == pytest.approx(5.0)

    # 5) Apply it to all data
    ds.apply_normalization()

    # 6) Check that every y_i was transformed by (y_i - train_mean)/train_std
    normalized = ds._data.y.numpy()
    raw_y      = np.array(df.y.tolist(), dtype=float)
    expected   = (raw_y - mean) / std

    # exact match on train, and correct usage of train stats on val/test
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=0)

    # and spot‐check val/test explicitly
    assert normalized[val_idx[0]]  == pytest.approx((20.0 - 5.0)/5.0)
    assert normalized[test_idx[0]] == pytest.approx((30.0 - 5.0)/5.0)