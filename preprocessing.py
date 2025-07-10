import numpy as _np
from rdkit.Chem.Scaffolds import MurckoScaffold

class StandardScaler:
    """
    Standardize target values: fit to data to compute mean and std,
    transform data to zero mean and unit variance, and inverse transform.
    Supports numpy arrays and torch tensors.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, y):
        """Compute mean and std from y."""
        # accept torch tensor or numpy array
        if hasattr(y, 'numpy'):
            import torch
            y_arr = y.detach().cpu()
            self.mean_ = float(y_arr.mean())
            self.std_ = float(y_arr.std(unbiased=False))
        else:
            import numpy as _np
            self.mean_ = float(_np.mean(y))
            self.std_ = float(_np.std(y))
        return self

    def transform(self, y):
        """Standardize y using the fitted mean and std."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler not fitted yet")
        # preserve type
        if hasattr(y, 'numpy'):
            import torch
            return (y - self.mean_) / self.std_
        else:
            import numpy as _np
            return (y - self.mean_) / self.std_

    def inverse_transform(self, y_scaled):
        """Invert the standardization."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler not fitted yet")
        if hasattr(y_scaled, 'numpy'):
            import torch
            return y_scaled * self.std_ + self.mean_
        else:
            import numpy as _np
            return y_scaled * self.std_ + self.mean_

def generate_scaffold(smiles: str) -> str:
    """
    Generate a Murcko scaffold from a SMILES string.
    Returns the canonical SMILES of the scaffold.
    """
    try:
        mol = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)
        return mol
    except Exception as e:
        print(f"Error generating scaffold for {smiles}: {e}")
        return None

def _safe_scaffold(smiles: str, idx: int) -> str:
    """Return a unique scaffold id; fallback to per-molecule token."""
    s = generate_scaffold(smiles)
    return s if s else f"_noscaffold_{idx}"

def scaffold_cross_validation(df, n_splits=5, seed=42):
    """
    Perform scaffold-based cross-validation splits.
    Returns a list of (train_df, test_df) tuples.
    """
    _np.random.seed(seed)
    # group indices by scaffold
    scaffolds = {}
    for idx, smiles in enumerate(df['smiles']):
        sc = _safe_scaffold(smiles, idx)
        scaffolds.setdefault(sc, []).append(idx)
    # sort scaffold groups by descending size
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    # assign each group to a fold in round-robin
    folds = [[] for _ in range(n_splits)]
    for i, grp in enumerate(groups):
        folds[i % n_splits].extend(grp)
    # build train/test splits
    all_idx = set(range(len(df)))
    splits = []
    for k in range(n_splits):
        test_idx = set(folds[k])
        train_idx = list(all_idx - test_idx)
        test_idx = list(test_idx)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)
        splits.append((train_df, test_df))
    return splits

def scaffold_cross_validation_repeated(df, n_splits=5, n_repeats=5, seed=42):
    """
    Perform repeated scaffold-based cross-validation.
    Returns a list of lists: each entry is the list of (train_df, test_df) for one repeat.
    """
    all_splits = []
    for r in range(n_repeats):
        folds = scaffold_cross_validation(df, n_splits=n_splits, seed=seed + r)
        all_splits.append(folds)
    return all_splits

def scaffold_split_indices(df, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Compute train/val/test index lists by scaffold splitting the DataFrame df.
    Returns (train_idx, valid_idx, test_idx).
    """
    _np.random.seed(seed)
    scaffolds = {}
    for idx, smiles in enumerate(df['smiles']):
        sc = _safe_scaffold(smiles, idx)
        scaffolds.setdefault(sc, []).append(idx)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(df)
    n_valid = int(n * valid_ratio)
    n_test = int(n * test_ratio)
    train_idx, valid_idx, test_idx = [], [], []
    for grp in groups:
        if len(train_idx) + len(grp) <= n - n_valid - n_test:
            train_idx.extend(grp)
        elif len(valid_idx) + len(grp) <= n_valid:
            valid_idx.extend(grp)
        else:
            test_idx.extend(grp)
    return train_idx, valid_idx, test_idx

def scaffold_split_indices_repeated(df, n_splits=5, n_repeats=5, seed=42):
    """
    Generate repeated scaffold-based splits returning index lists.
    Returns list of repeats, each repeat is list of (train_idx, valid_idx, test_idx).
    """
    all_splits = []
    for r in range(n_repeats):
        repeats = []
        # create n_splits distinct test splits by varying seed
        for k in range(n_splits):
            tr, va, te = scaffold_split_indices(df, valid_ratio=0.1, test_ratio=0.1, seed=seed + r * n_splits + k)
            repeats.append((tr, va, te))
        all_splits.append(repeats)
    return all_splits

def random_split_indices(df, valid_ratio=0.1, test_ratio=0.1, seed=42) -> tuple[list[int], list[int], list[int]]:
    """
    Produce three disjoint index lists (train, valid, test) by *randomly*
    sampling rows from `df` with the supplied ratios.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have the same length as the dataset.
    valid_ratio, test_ratio : float
        Fractions of the whole dataset to allocate to the validation and
        test sets, respectively.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    train_idx, valid_idx, test_idx : list[int]
        Lists of row indices into `df`.
    """
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError("valid_ratio + test_ratio must be < 1.0")

    _np.random.seed(seed)

    all_idx = _np.arange(len(df))
    _np.random.shuffle(all_idx)

    n = len(df)
    n_valid = int(round(n * valid_ratio))
    n_test  = int(round(n * test_ratio))

    valid_idx = all_idx[:n_valid].tolist()
    test_idx  = all_idx[n_valid : n_valid + n_test].tolist()
    train_idx = all_idx[n_valid + n_test :].tolist()

    return train_idx, valid_idx, test_idx

def random_split_indices_repeated(df, n_splits=5, n_repeats=5, seed=42) -> list[tuple[list[int], list[int], list[int]]]:
    """
    Generate repeated random splits returning index lists.
    Returns list of repeats, each repeat is list of (train_idx, valid_idx, test_idx).
    """
    all_splits = []
    for r in range(n_repeats):
        repeats = []
        # create n_splits distinct test splits by varying seed
        for k in range(n_splits):
            tr, va, te = random_split_indices(df, valid_ratio=0.1, test_ratio=0.1, seed=seed + r * n_splits + k) # deterministic variation
            repeats.append((tr, va, te))
        all_splits.append(repeats)
    return all_splits

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances
import numpy as np

def featurize_rdkit_fp(smiles_list, radius=2, nbits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nbits))  # fallback
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(np.array(fp))
    return np.stack(fps)

def kennard_stone_split(X, k, seed=42):
    np.random.seed(seed)
    N = len(X)
    selected = []

    # Step 1: Choose 2 samples with max distance
    D = pairwise_distances(X)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected.extend([i, j])

    # Step 2: Iteratively add the sample farthest from selected
    while len(selected) < k:
        remaining = list(set(range(N)) - set(selected))
        min_dists = np.min(D[remaining][:, selected], axis=1)
        farthest = remaining[np.argmax(min_dists)]
        selected.append(farthest)

    return selected

def kennard_stone_cv(df, n_splits=5, seed=42):
    smiles_list = df["smiles"].tolist()
    X = featurize_rdkit_fp(smiles_list)
    N = len(df)
    selected = kennard_stone_split(X, N, seed=seed)

    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(selected):
        folds[i % n_splits].append(idx)

    splits = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = list(set(range(N)) - set(val_idx))
        splits.append((df.iloc[train_idx], df.iloc[val_idx]))

    return splits

def kennard_stone_cross_validation_repeated(df, n_splits=5, n_repeats=5, seed=42):
    all_splits = []
    for r in range(n_repeats):
        splits = kennard_stone_cv(df, n_splits=n_splits, seed=seed + r)
        all_splits.append(splits)
    return all_splits