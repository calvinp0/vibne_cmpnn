import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
import shutil
from rdkit.Chem.Scaffolds import MurckoScaffold
import argparse
import warnings
import hashlib
from typing import Union, Sequence, Any, Callable, Optional
from pathlib import Path
import math
from global_feat import CompositeGlobalFeaturizer
# import pathlib
import pathlib

# suppress warning about direct data access in torch_geometric InMemoryDataset
warnings.filterwarnings('ignore', \
    message='It is not recommended to directly access the internal storage format `data` of an \'InMemoryDataset\'.*',
    category=UserWarning,
    module='torch_geometric.data.in_memory_dataset')

# feature mappings
STEREO = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}


def onek_encoding_unk(value: int, choices: list[int]) -> list[int]:
    """One-hot with an extra 'unknown' slot at the end."""
    enc = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else -1
    enc[idx] = 1
    return enc

def atom_features(atom: Chem.Atom) -> np.ndarray:
    """133-D atom feature vector identical to Chemprop."""
    feats = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, list(range(100))) +       # 101
        onek_encoding_unk(atom.GetTotalDegree(),       [0,1,2,3,4,5]) +      # 7
        onek_encoding_unk(atom.GetFormalCharge(),      [-2,-1,0,1,2]) +      # 6
        onek_encoding_unk(int(atom.GetChiralTag()),    [0,1,2,3]) +          # 5
        onek_encoding_unk(int(atom.GetTotalNumHs()),   [0,1,2,3,4]) +        # 6
        onek_encoding_unk(int(atom.GetHybridization()),
                          [Chem.rdchem.HybridizationType.SP,
                           Chem.rdchem.HybridizationType.SP2,
                           Chem.rdchem.HybridizationType.SP3,
                           Chem.rdchem.HybridizationType.SP3D,
                           Chem.rdchem.HybridizationType.SP3D2]) +           # 6
        [1 if atom.GetIsAromatic() else 0] +                                 # 1
        [atom.GetMass() * 0.01]                                              # 1
    )
    return np.array(feats, dtype=float) 

def bond_features(bond):
    feats = []
    # bond type one-hot (size 4)
    bt = bond.GetBondType()
    one_bt = [0] * 4
    if bt == Chem.rdchem.BondType.SINGLE:
        one_bt[0] = 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        one_bt[1] = 1
    elif bt == Chem.rdchem.BondType.TRIPLE:
        one_bt[2] = 1
    else:
        one_bt[3] = 1
    feats.extend(one_bt)
    # conjugation (1)
    feats.append(1 if bond.GetIsConjugated() else 0)
    # in ring (1)
    feats.append(1 if bond.IsInRing() else 0)
    # stereo one-hot (size 6)
    st = bond.GetStereo()
    one_st = [0] * 6
    one_st[STEREO.get(st, 0)] = 1
    feats.extend(one_st)
    return np.array(feats, dtype=float)


def featurize_to_Data(mol_or_smiles: str, atom_messages: bool = False) -> Data:
    """
    Converts a SMILES string to a PyG `Data` graph with attributes:
        x            (num_atoms , A_dim)
        edge_index   (2, num_edges*2)
        edge_attr    (num_edges*2 , E_dim or A_dim+E_dim)
    No target `y` is attached here – caller decides.
    Returns `None` if RDKit fails to parse.
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    if mol is None:
        return None

    # ---------- atom features ----------
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.vstack(atom_feats), dtype=torch.float)

    # ---------- bond features (directed edges) ----------
    e_idx, e_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        if atom_messages:
            m1 = m2 = bf                               #  edge-only message
        else:
            m1 = np.concatenate([atom_feats[i], bf])   #  atom ⊕ bond
            m2 = np.concatenate([atom_feats[j], bf])
        e_idx += [[i, j], [j, i]]
        e_attr += [m1, m2]

    if len(e_attr) == 0:
        # no bonds  →  empty tensors with correct dims
        feat_dim = len(e_attr[0]) if e_attr else (len(atom_feats[0]) + 12)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, feat_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(e_idx, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(np.vstack(e_attr), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ================================
#  Helper functions for extra‐feat processing
# ================================
def rbf_expand(
    values: np.ndarray,
    num_centers: int,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """
    Expand a 1D array of radii into RBF features.  
    Centers are linear from r_min to r_max.
    """
    # Create evenly spaced centers between r_min and r_max
    centers = np.linspace(r_min, r_max, num_centers)
    # Choose a bandwidth so that neighboring RBFs have moderate overlap
    # For a gaussian: σ = (max−min)/(num_centers−1)
    sigma = (r_max - r_min) / (num_centers - 1)
    diff = values.reshape(-1, 1) - centers.reshape(1, -1)
    return np.exp(-0.5 * (diff / sigma) ** 2)


def dihedral_to_sin_cos(dihedral_vals: np.ndarray) -> np.ndarray:
    """
    Convert dihedral angles (in degrees) to [sin, cos] pair.  
    Input is shape (N,), output is (N, 2).
    """
    radians = np.deg2rad(dihedral_vals)
    return np.stack([np.sin(radians), np.cos(radians)], axis=1)


def normalize_angle(
    angle_vals: np.ndarray,
    a_min: float,
    a_max: float,
) -> np.ndarray:
    """
    Linearly scale an angle (in degrees) to [0, 1], based on training min/max.
    """
    return (angle_vals - a_min) / (a_max - a_min + 1e-12)



class CMPNNDataset(InMemoryDataset):
    def __init__(self, root, csv_file, atom_messages=False, transform=None, pre_transform=None, force_reload=False,
                 weights_only=False):
        self.csv_file = csv_file
        self.atom_messages = atom_messages
        # compute SHA-1 of the raw CSV to namespace cache
        raw_path = os.path.join(root, 'raw', csv_file)
        with open(raw_path, 'rb') as f:
            data = f.read()
        self.csv_hash = hashlib.sha1(data + str(atom_messages).encode()).hexdigest()
        # optionally clear this fold's processed cache when forcing reload
        proc_dir = os.path.join(root, 'processed', self.csv_hash)
        if force_reload:
            shutil.rmtree(proc_dir, ignore_errors=True)
        # process raw data and initialize (caching applies per-hash)
        super().__init__(root, transform, pre_transform)
        # load processed data into internal buffer (use _data to avoid UserWarning)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=weights_only)
        self.raw_y = self.data.y.clone()
        # store raw targets; normalization deferred per fold
        self.raw_y = self._data.y.clone()
        self.mean = None
        self.std = None

    def compute_normalization(self, indices):
        """
        Fit normalization parameters on raw_y at given train indices.
        """
        y = self.raw_y[indices]
        mean = y.mean()
        std = y.std(unbiased=False)
        self.mean = mean.item()
        self.std = std.item()
        return self.mean, self.std

    def apply_normalization(self):
        if self.mean is None or self.std is None:
            raise ValueError("…")
        # write into the *actual* storage
        self.data.y = (self.raw_y - self.mean) / self.std
    
    def inverse_normalization(self):
        if self.mean is None or self.std is None:
            raise ValueError("…")
        # write into the *actual* storage
        self.data.y = (self.data.y * self.std) + self.mean
    
    def __len__(self):
        # return the number of graphs in the dataset
        return len(self.data.y)
    

    @property
    def raw_file_names(self):
        # the raw SMILES CSV is expected under raw_dir
        return [self.csv_file]

    @property
    def processed_dir(self):
        # separate processed folders per CSV hash
        return os.path.join(self.root, 'processed', self.csv_hash)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # raw data already in place
        pass

    def process(self):
        df = pd.read_csv(os.path.join(self.raw_dir, self.csv_file))
        data_list = []
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                continue
            # atom features
            atom_feats = [atom_features(a) for a in mol.GetAtoms()]
            x = torch.tensor(np.vstack(atom_feats), dtype=torch.float)
            # bond index and features (directed)
            edge_idx = []
            edge_attr = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = bond_features(bond)
                if self.atom_messages:
                    m1 = m2 = bf
                else:
                    m1 = np.concatenate([atom_feats[i], bf])
                    m2 = np.concatenate([atom_feats[j], bf])
                edge_idx += [[i, j], [j, i]]
                edge_attr += [m1, m2]
            # if no bonds, create empty tensors to avoid vstack error
            if len(edge_attr) == 0:
                feat_dim = 12 if self.atom_messages else atom_feats[0].shape[0] + 12
                edge_attr = torch.empty((0, feat_dim), dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(np.vstack(edge_attr), dtype=torch.float)
            # target
            y = torch.tensor([row['y']], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_atom_features(self) -> int:
        """Dimensionality of the node feature vectors (the size of xᵢ)."""
        # self.data.x is all node features stacked, shape [total_nodes, num_feats]
        return int(self.data.x.size(1))

    @property
    def num_bond_features(self) -> int:
        """Dimensionality of the edge feature vectors (the size of eᵢⱼ)."""
        # self.data.edge_attr is all edge features stacked, shape [total_edges, num_feats]
        return int(self.data.edge_attr.size(1))

    @property
    def num_targets(self) -> int:
        """Number of target values per graph."""
        # self.data.y is the concatenated graph targets: 
        # if shape = [num_graphs] → scalar target, so return 1
        # if shape = [num_graphs, T] → multi-task targets
        y = self.data.y
        if y.dim() == 1:
            return 1
        else:
            return int(y.size(1))


class MultiCMPNNDataset(InMemoryDataset):
    """
    Dataset holding pairs of molecules sharing the same target y.
    CSV must have columns 'smiles1', 'smiles2', and 'y'.
    """
    def __init__(self, root, csv_file, atom_messages=False, transform=None,
                 pre_transform=None, force_reload=False):
        self.csv_file = csv_file
        self.atom_messages = atom_messages
        # compute cache hash
        raw_path = os.path.join(root, 'raw', csv_file)
        with open(raw_path, 'rb') as f:
            data = f.read()
        self.csv_hash = hashlib.sha1(data + str(atom_messages).encode()).hexdigest()
        # clear cache if forced
        proc_dir = os.path.join(root, 'processed', self.csv_hash)
        if force_reload:
            shutil.rmtree(proc_dir, ignore_errors=True)
        super().__init__(root, transform, pre_transform)
        # load processed data
        self.pairs: list[tuple[Data, Data]] = torch.load(self.processed_paths[0], weights_only=False)
        self.raw_y = torch.tensor([p[0].y.item() for p in self.pairs])
        self.mean  = None
        self.std   = None

    @property
    def raw_file_names(self):
        return [self.csv_file]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.csv_hash)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        df = pd.read_csv(os.path.join(self.raw_dir, self.csv_file))
        data_pairs = []
        for idx, row in df.iterrows():
            g1 = featurize_to_Data(row["smiles1"], self.atom_messages)
            g2 = featurize_to_Data(row["smiles2"], self.atom_messages)
            # attach the *same* y to both graphs (simplest)
            g1.y = g2.y = torch.tensor([row["y"]], dtype=torch.float)
            data_pairs.append((g1, g2))
        torch.save(data_pairs, self.processed_paths[0])

    def compute_normalization(self, indices: list[int]):
        """
        Fit μ, σ from the training rows.
        `indices` index *pairs* (CSV rows), so we just look at the first graph
        of each pair to get one y per row.
        """
        ys = torch.tensor([self.pairs[i][0].y.item() for i in indices])
        self.mean = ys.mean().item()
        self.std  = ys.std(unbiased=False).item()
        return self.mean, self.std

    def apply_normalization(self):
        """
        Apply (y-μ)/σ to every graph in every pair.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("compute_normalization must be called first")

        for g1, g2 in self.pairs:
            g1.y = g2.y = (g1.y - self.mean) / self.std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        g1, g2 = self.pairs[idx]
        if self.transform:
            g1 = self.transform(g1)
            g2 = self.transform(g2)
        return g1, g2

    @property
    def num_atom_features(self) -> int:
        """Length of every node-feature vector."""
        return self.pairs[0][0].x.size(1)          # first graph of first pair

    @property
    def num_bond_features(self) -> int:
        """Length of every edge-feature vector."""
        return self.pairs[0][0].edge_attr.size(1)

    @property
    def num_targets(self) -> int:
        """Number of prediction targets per sample (almost always 1)."""
        return self.pairs[0][0].y.numel()
    
class MultiCMPNNDatasetSDF(InMemoryDataset):
    def __init__(
        self,
        root: str,
        sdf_files: Union[str, Path, Sequence[Union[str, Path]]],
        target_df: Union[str, Path, pd.DataFrame],
        target_cols: Sequence[str],
        target_types: dict[str, str],
        atom_messages: bool = False,
        global_featurizer: Optional[CompositeGlobalFeaturizer] = None,
        keep_hs: bool = True,
        sanitize: bool = False,
        input_type: tuple[str, ...] = ('r1h', 'r2h'),
        transform=None,
        pre_transform=None,
        force_reload=False,
        prune_value: float | None = None,
        # ── NEW parameters for extra‐feats:
        atom_extra_feats: Optional[Union[str, Path, pd.DataFrame]] = None,
        rbf_num_centers: int = 16,
    ):
        self.atom_messages       = atom_messages
        self.keep_hs             = keep_hs
        self.sanitize            = sanitize
        self.input_type          = tuple(input_type)
        self.target_cols         = list(target_cols)
        self.target_types        = target_types
        self.global_featurizer   = global_featurizer
        self.rbf_num_centers     = rbf_num_centers

        # identify which targets are continuous vs. periodic
        self.cont_idx = [
            i for i, c in enumerate(self.target_cols)
            if self.target_types[c] == "continuous"
        ]
        self.per_idx = [
            i for i, c in enumerate(self.target_cols)
            if self.target_types[c] == "periodic"
        ]

        # ── load or parse target_df
        if isinstance(target_df, (str, Path)):
            target_path = Path(target_df).expanduser().resolve()
            if target_path.suffix in {".csv", ".gz"}:
                target_df = pd.read_csv(target_path)
            elif target_path.suffix in {".parquet"}:
                target_df = pd.read_parquet(target_path)
            else:
                raise ValueError(f"Unsupported target file type: {target_path}")
        self.target_df: pd.DataFrame = target_df.reset_index(drop=True)

        if not all(col in self.target_df.columns for col in self.target_cols):
            missing = [c for c in self.target_cols if c not in self.target_df.columns]
            raise KeyError(f"Missing target columns: {missing}")

        # ── load or parse atom_extra_feats (if given)
        if atom_extra_feats is None:
            self.raw_atom_extra_feats = None
        else:
            if isinstance(atom_extra_feats, (str, Path)):
                path = Path(atom_extra_feats).expanduser().resolve()
                if path.suffix in {".csv", ".gz"}:
                    self.raw_atom_extra_feats = pd.read_csv(path)
                elif path.suffix in {".parquet"}:
                    self.raw_atom_extra_feats = pd.read_parquet(path)
                else:
                    raise ValueError(f"Unsupported extra‐feat file type: {path}")
            else:
                # assume already a DataFrame
                self.raw_atom_extra_feats = atom_extra_feats.copy()
        # we will fill `self.atom_extra_feats_out` later, once train‐split is known
        self.atom_extra_feats_out = None

        # ── now build a cache hash that includes or excludes the extra‐feat file’s name
        h = hashlib.sha1()
        # hash all SDF paths + mtimes + target_df content + target columns + flags
        if isinstance(sdf_files, (str, Path)):
            sdf_files = [sdf_files]
        expanded = []
        for p in sdf_files:
            pth = Path(p).resolve()
            if pth.is_dir():
                expanded += sorted(pth.glob("*.sdf"))
            else:
                expanded.append(pth)
        if len(expanded) == 0:
            raise FileNotFoundError("No .sdf files found in the given path(s).")
        self.sdf_paths = expanded
        for pth in self.sdf_paths:
            h.update(str(pth).encode())
            h.update(str(os.path.getmtime(pth)).encode())

        # hash the target DataFrame content
        h.update(pd.util.hash_pandas_object(self.target_df, index=False).values)
        h.update(",".join(self.target_cols).encode())
        h.update(str(atom_messages).encode())
        h.update(str(keep_hs).encode())
        h.update(str(sanitize).encode())
        h.update(",".join(self.input_type).encode())

        # if there's an extra‐feat CSV, hash its file name (so changes in that file name
        # will create a different `processed/` subfolder)
        if self.raw_atom_extra_feats is not None:
            # note: Path(atom_extra_feats) is only valid if it was a string
            if isinstance(atom_extra_feats, (str, Path)):
                h.update(pathlib.Path(atom_extra_feats).name.encode())

        self.cache_hash = h.hexdigest()
        self._proc_dir  = Path(root) / "processed" / self.cache_hash
        if force_reload and self._proc_dir.exists():
            shutil.rmtree(self._proc_dir, ignore_errors=True)
        os.makedirs(self._proc_dir, exist_ok=True)

        # ── call super().__init__, which will either find or remake `processed/multi_sdf_data.pt`
        super().__init__(root, transform, pre_transform)

        # ── load every graph‐pair from disk (each pair = two PyG Data objects)
        all_pairs = torch.load(self.processed_paths[0], weights_only=False)

        # ── prune out any pair whose Y contains prune_value
        if prune_value is not None:
            pruned = []
            for g1, g2 in all_pairs:
                if not (g1.y == prune_value).any():
                    pruned.append((g1, g2))
            self.pairs = pruned
        else:
            self.pairs = all_pairs

        # raw_y is only used for later statistics
        self.raw_y = torch.stack([p[0].y.clone() for p in self.pairs], dim=0)
        self.mean  = self.std = None

    @property
    def raw_file_names(self) -> list[str]:
        return [p.name for p in self.sdf_paths]

    @property
    def processed_dir(self) -> str:
        return str(self._proc_dir)

    @property
    def processed_file_names(self) -> list[str]:
        return ["multi_sdf_data.pt"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        g1, g2 = self.pairs[idx]
        if self.transform:
            g1 = self.transform(g1)
            g2 = self.transform(g2)
        return g1, g2

    def process(self):
        """
        Exactly unchanged from before: read every SDF, group by 'reaction', build
        two PyG Data graphs (g1, g2) for each reaction, attach base‐Y to each, and
        save the list of (g1, g2) into processed_paths[0].
        """
        data_pairs: list[tuple[Data, Data]] = []
        for sdf_path in self.sdf_paths:
            suppl = Chem.SDMolSupplier(
                str(sdf_path),
                removeHs = not self.keep_hs,
                sanitize  = self.sanitize,
            )

            buckets: dict[str, list[Chem.Mol]] = {}
            for mol in suppl:
                if mol is None:
                    continue
                if not mol.HasProp("type") or mol.GetProp("type") not in self.input_type:
                    continue
                if not mol.HasProp("reaction"):
                    warnings.warn(f"{sdf_path.name}: molecule missing 'reaction' tag")
                    continue
                r = mol.GetProp("reaction")
                buckets.setdefault(r, []).append(mol)

            for rxn, mols in buckets.items():
                if len(mols) != 2:
                    warnings.warn(f"{rxn}: expected exactly 2 mols, got {len(mols)} – skipped")
                    continue

                # build base Data objects (no extra‐feats yet)
                g1 = featurize_to_Data(mols[0], self.atom_messages)
                g2 = featurize_to_Data(mols[1], self.atom_messages)

                # attach Y to each
                row = self.target_df[self.target_df["rxn"] == rxn]
                if row.empty:
                    warnings.warn(f"No target row for reaction '{rxn}' – skipped")
                    continue
                y_vals = row.iloc[0][self.target_cols].to_numpy(dtype="float32")
                y = torch.tensor(y_vals, dtype=torch.float32)
                g1.y = g2.y = y

                if g1 is None or g2 is None:
                    warnings.warn(f"Failed to featurize {rxn} – skipped")
                    continue

                if self.pre_transform:
                    g1 = self.pre_transform(g1)
                    g2 = self.pre_transform(g2)

                if self.global_featurizer is not None:
                    try:
                        smiles1 = mols[0].GetProp("smiles") if mols[0].HasProp("smiles") else Chem.MolToSmiles(mols[0])
                        smiles2 = mols[1].GetProp("smiles") if mols[1].HasProp("smiles") else Chem.MolToSmiles(mols[1])
                        gf1 = self.global_featurizer(smiles1)
                        gf2 = self.global_featurizer(smiles2)
                        g1.global_features = torch.tensor(gf1, dtype=torch.float32).view(1, -1)
                        g2.global_features = torch.tensor(gf2, dtype=torch.float32).view(1, -1)
                    except Exception as e:
                        warnings.warn(f"Failed to featurize global features for {rxn}: {e}")
                        continue

                g1.name     = mols[0].GetProp("type") if mols[0].HasProp("type") else f"{rxn}_mol1"
                g2.name     = mols[1].GetProp("type") if mols[1].HasProp("type") else f"{rxn}_mol2"
                g1.smiles = Chem.MolToSmiles(mols[0])
                g2.smiles = Chem.MolToSmiles(mols[1])
                g1.reaction = g2.reaction = rxn

                data_pairs.append((g1, g2))

        if len(data_pairs) == 0:
            raise RuntimeError("No valid (mol1, mol2, target) pairs found!")

        torch.save(data_pairs, self.processed_paths[0])

    # ────────────────────────────────────────────────────────────────────────────
    def compute_normalization(self, indices: list[int]):
        """
        Exactly your old code that only standardizes the continuous Y columns and
        turns periodic Y columns into sin/cos.  Nothing about X or extra‐feats here.
        """
        ys = torch.stack([ self.pairs[i][0].y for i in indices ])
        cont = ys[:, self.cont_idx] if self.cont_idx else torch.zeros((ys.size(0), 0))
        self.cont_mean = cont.mean(dim=0) if cont.numel() > 0 else torch.zeros_like(cont[0])
        self.cont_std  = (
            cont.std(dim=0, unbiased=False).clamp(min=1e-12)
            if cont.numel() > 0 else torch.ones_like(cont[0])
        )

        per_deg = ys[:, self.per_idx] if self.per_idx else torch.zeros((ys.size(0), 0))
        per_rad = per_deg * (math.pi / 180)
        if per_rad.numel() > 0:
            sin_sum = per_rad.sin().mean(dim=0)
            cos_sum = per_rad.cos().mean(dim=0)
            self.per_mean_dir = torch.atan2(sin_sum, cos_sum)
        else:
            self.per_mean_dir = torch.zeros(0)

        # do NOT mutate X here—only store cont_mean, cont_std, per_mean_dir.
        self.mean = self.cont_mean
        self.std  = self.cont_std
        return self.cont_mean, self.cont_std, self.per_mean_dir

    # ────────────────────────────────────────────────────────────────────────────
    def apply_normalization(self):
        """
        Take each (g1, g2) in self.pairs, look at g1.y, do exactly:
          - y[cont_idx] = (y[cont_idx] - cont_mean)/cont_std
          - build sin/cos for periodic
          - set g1.y = g2.y = new y vector
        (no changes to X here.)
        """
        if self.cont_mean is None or self.per_mean_dir is None:
            raise RuntimeError("Must call compute_normalization first.")

        for (g1, g2) in self.pairs:
            y = g1.y.clone().view(-1)  # raw cont + periodic (deg)
            if self.cont_idx:
                y[self.cont_idx] = (y[self.cont_idx] - self.cont_mean) / self.cont_std
            if self.per_idx:
                pd = y[self.per_idx] * (math.pi / 180)  # to radians
                shifted = ((pd - self.per_mean_dir) + math.pi) % (2 * math.pi) - math.pi
                sinv = shifted.sin()
                cosv = shifted.cos()
                new_per = torch.stack([sinv, cosv], dim=1).view(-1)
            else:
                new_per = torch.zeros(0)

            # rebuild y = [normalized continuous ∥ new_per] 
            if self.cont_idx:
                cont_part = y[self.cont_idx]
            else:
                cont_part = torch.zeros(0)
            y_final = torch.cat([cont_part, new_per], dim=0)
            g1.y = g2.y = y_final.unsqueeze(0)

    # ────────────────────────────────────────────────────────────────────────────
    def attach_atom_extra_features(self, train_indices: list[int]):
        """
        (1) Figure out r_min/r_max, a_min/a_max from only those extra‐feat rows whose
        rxn_id appears in self.pairs[i][0].reaction for i in train_indices.
        (2) Build new DataFrame `self.atom_extra_feats_out` that holds, for every row in
            raw_atom_extra_feats:
              • radius → radius_rbf_{0..rbf_num_centers-1}
              • dihedral → [dihedral_sin, dihedral_cos]
              • angle → [angle_norm]
            plus any leftover boolean columns.  We re‐attach “rxn_id”, “mol_type”,
            “focus_atom_idx” so we can filter later.
        (3) Finally, for every graph in self.pairs, splice the extra vector into `Data.x`.
            If no row exists for that (rxn_id, mol_type), fill with zeros for all atoms.
        """
        if self.raw_atom_extra_feats is None:
            # nothing to do
            return

        # 1) which rxn_ids are in the training set?
        train_rxns = { self.pairs[i][0].reaction for i in train_indices }

        afeats = self.raw_atom_extra_feats
        train_mask = afeats["rxn_id"].isin(train_rxns)
        train_subset = afeats.loc[train_mask]
        if train_subset.shape[0] == 0:
            raise RuntimeError("No extra‐feat rows match the training rxn_ids.")

        # compute r_min/r_max on training subset’s “radius”
        self.r_min, self.r_max = (
            float(train_subset["radius"].min()),
            float(train_subset["radius"].max()),
        )
        # compute a_min/a_max on “angle”
        self.a_min, self.a_max = (
            float(train_subset["angle"].min()),
            float(train_subset["angle"].max()),
        )


        # 1.a) Need to Fiill Nans -> 0 for radius, angle, dihedral
        afeats["radius"]   = afeats["radius"].fillna(0.0)
        afeats["angle"]    = afeats["angle"].fillna(0.0)
        afeats["dihedral"] = afeats["dihedral"].fillna(0.0)
        # 2) Expand **all** rows of raw_atom_extra_feats → new columns
        df = afeats.copy()
        radii = df["radius"].astype(float).values
        rbf_mat = rbf_expand(
            radii,
            num_centers = self.rbf_num_centers,
            r_min        = self.r_min,
            r_max        = self.r_max,
        )
        for i in range(self.rbf_num_centers):
            df[f"radius_rbf_{i}"] = rbf_mat[:, i]

        dihed = df["dihedral"].astype(float).values
        dc = dihedral_to_sin_cos(dihed)
        df["dihedral_sin"] = dc[:, 0]
        df["dihedral_cos"] = dc[:, 1]

        ang = df["angle"].astype(float).values
        df["angle_norm"] = normalize_angle(ang, a_min=self.a_min, a_max=self.a_max)

        drop_cols = [
            "rxn_id", "mol_type", "focus_atom_idx",
            "path", "radius", "angle", "dihedral", "focus_atom_symbol",
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        # re‐introduce the three key columns for lookup
        df["rxn_id"]        = afeats["rxn_id"]
        df["mol_type"]      = afeats["mol_type"]
        df["focus_atom_idx"]= afeats["focus_atom_idx"]

        # reorder so that keys come first
        cols_ordered = ["rxn_id", "mol_type", "focus_atom_idx"] + [
            c for c in df.columns if c not in {"rxn_id","mol_type","focus_atom_idx"}
        ]
        df = df[cols_ordered].reset_index(drop=True)

        self.atom_extra_feats_out = df

        # 3) Now attach to each graph’s x:
        for (g1, g2) in self.pairs:
            self._attach_extras_to_graph(g1)
            self._attach_extras_to_graph(g2)

    def _attach_extras_to_graph(self, data_obj: Data):
        rxn_id   = data_obj.reaction
        mol_type = data_obj.name

        # filter exactly that row for (rxn_id, mol_type)
        df = self.atom_extra_feats_out
        mask = (df["rxn_id"] == rxn_id) & (df["mol_type"] == mol_type)
        sub_df = df.loc[mask]

        num_atoms = data_obj.x.size(0)
        extra_cols = [c for c in df.columns if c not in {"rxn_id","mol_type","focus_atom_idx"}]
        extra_dim  = len(extra_cols)

        if sub_df.shape[0] == 0:
            extras = torch.zeros((num_atoms, extra_dim), dtype=data_obj.x.dtype)
        else:
            row       = sub_df.iloc[0]
            focus_idx = int(row["focus_atom_idx"])
            atom_vals = row[extra_cols].to_numpy(dtype=float)
            extras    = torch.zeros((num_atoms, extra_dim), dtype=data_obj.x.dtype)
            if 0 <= focus_idx < num_atoms:
                extras[focus_idx, :] = torch.tensor(atom_vals, dtype=data_obj.x.dtype)

        data_obj.x = torch.cat([data_obj.x, extras], dim=1)

    # ────────────────────────────────────────────────────────────────────────────
    @property
    def num_atom_features(self) -> int:
        # returns the current x‐dimension (133 + extra_dim), after extras are attached
        return self.pairs[0][0].x.size(1)

    @property
    def num_bond_features(self) -> int:
        return self.pairs[0][0].edge_attr.size(1)

    @property
    def num_targets(self) -> int:
        return self.pairs[0][0].y.numel()

    @property
    def global_feature_dim(self) -> int:
        gf = self.pairs[0][0].global_features
        if isinstance(gf, torch.Tensor):
            return gf.numel()
        elif isinstance(gf, np.ndarray):
            return gf.size
        raise TypeError(f"Unsupported global_features type: {type(gf)}")


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from typing import Callable, Sequence, Tuple
from utils_paired import collate_pairs

class CMPNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # core args to build your full dataset
        root: str,
        sdf_files,
        target_df,
        target_cols: Sequence[str],
        target_types: dict[str,str],
        # transforms
        train_transform,
        splitter: Callable[[Any], Tuple[Sequence[int],Sequence[int],Sequence[int]]],
        val_transform=None,
        test_transform=None,
        # your splitter: returns three index lists
        batch_size: int = 32,
        num_workers: int = 4,
        # ratios or any args you need are captured by a lambda
        atom_messages: bool = False,
        keep_hs: bool = True,
        sanitize: bool = False,
        input_type: tuple[str, ...] = ('r1h', 'r2h'),
        force_reload: bool = False,
        prune_value: float | None = None,
    ):
        super().__init__()
        self.root            = root
        self.sdf_files       = sdf_files
        self.target_df       = target_df
        self.target_cols     = list(target_cols)
        self.target_types    = target_types
        self.train_transform = train_transform
        self.val_transform   = val_transform   or train_transform
        self.test_transform  = test_transform  or self.val_transform
        self.splitter        = splitter
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.atom_messages   = atom_messages
        self.keep_hs         = keep_hs
        self.sanitize        = sanitize
        self.input_type      = tuple(input_type)
        self.force_reload    = force_reload
        self.prune_value     = prune_value


    def setup(self, stage=None):
        # 1) build the *full* dataset once (without any Subset)
        full_ds = MultiCMPNNDatasetSDF(
            root=self.root,
            sdf_files=self.sdf_files,
            target_df=self.target_df,
            target_cols=self.target_cols,
            target_types=self.target_types,
            atom_messages=self.atom_messages,
            keep_hs=self.keep_hs,
            sanitize=self.sanitize,
            input_type=self.input_type,
            transform=None,          # we’ll apply transforms per‐split
            prune_value=self.prune_value,
        )

        # 2) split indices
        train_idx, val_idx, test_idx = self.splitter(full_ds)




        #Center psi1
        train_phis = full_ds.raw_y[train_idx, 0]    # shape: (len(train_idx),)
# if those are in degrees, convert to radians now:
        import math
        train_phis = train_phis * (math.pi / 180)
        sin_sum = torch.sin(train_phis).mean()
        cos_sum = torch.cos(train_phis).mean()
        mean_dir = torch.atan2(sin_sum, cos_sum)
        self.mean_dir = mean_dir
        for g1, g2 in full_ds.pairs:
            phi = g1.y[0]
            g1.y[0] = g2.y[0] = center_angle(phi, self.mean_dir) * (180 / math.pi)

        # 3) compute & apply normalization *only* on train samples
        full_ds.compute_normalization(train_idx)
        full_ds.apply_normalization()

        # 4) wrap in SubsetWithTransform so each split gets its own transform
        self.train_dataset = SubsetWithTransform(full_ds, train_idx, transform=self.train_transform)
        self.val_dataset   = SubsetWithTransform(full_ds,   val_idx, transform=self.val_transform)
        self.test_dataset  = SubsetWithTransform(full_ds,  test_idx, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pairs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pairs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pairs
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pairs
        )

from torch.utils.data import Subset

class SubsetWithTransform(Subset):
    """
    A Subset that applies a different `transform` to each sample.
    """
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        # get the pair (g1, g2) from the *full* dataset
        g1, g2 = self.dataset[self.indices[idx]]
        # apply *this* subset’s transform
        if self.transform:
            g1 = self.transform(g1)
            g2 = self.transform(g2)
        return g1, g2

    @property
    def num_targets(self):
        g1, _ = self[0]              # runs the transform once
        return g1.y.numel()

def center_angle(phi, mean_dir):
    # shift phi so mean_dir → 0, then wrap back into [−π, π]
    return ((phi - mean_dir) + np.pi) % (2*np.pi) - np.pi
