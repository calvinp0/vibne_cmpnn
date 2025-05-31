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
    """
    Dataset holding pairs of molecules sharing the same target y.
    SDF must have columns 'smiles1', 'smiles2', and 'y'.
    """
    def __init__(self, root: str, 
                 sdf_files: Union[str, Path, Sequence[Union[str, Path]]],
                 target_df: Union[str, Path, pd.DataFrame],
                 target_cols: Sequence[str],
                 target_types: dict[str, str],
                 atom_messages: bool = False, global_featurizer: Optional[CompositeGlobalFeaturizer] = None,
             keep_hs: bool = True,
             sanitize: bool = False,
             input_type: tuple[str, ...] = ('r1h', 'r2h'),
             transform=None, pre_transform=None, force_reload=False,
             prune_value: float | None = None):
        self.atom_messages = atom_messages
        self.keep_hs       = keep_hs
        self.sanitize      = sanitize
        self.input_type    = tuple(input_type)
        self.target_cols   = list(target_cols)
        self.target_types  = target_types
        self.global_featurizer = global_featurizer

        self.cont_idx = [i for i, c in enumerate(self.target_cols)
                          if self.target_types[c] == "continuous"]
        self.per_idx = [i for i, c in enumerate(self.target_cols)
                            if self.target_types[c] == "periodic"]

        if isinstance(sdf_files, (str, Path)):
            sdf_files = [sdf_files]

        expanded = []
        for p in sdf_files:
            p = Path(p).resolve()
            if p.is_dir():
                expanded += sorted(p.glob("*.sdf"))
            else:
                expanded.append(p)

        if len(expanded) == 0:
            raise FileNotFoundError("No .sdf files found in the given path(s).")
        self.sdf_paths   = expanded 
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
            missing = [c for c in self.target_cols if c not in self.target_df]
            raise KeyError(f"Missing target columns: {missing}")

        # compute cache hash
        h = hashlib.sha1()
        for p in self.sdf_paths:
            h.update(str(p).encode())
            h.update(str(os.path.getmtime(p)).encode())
        h.update(pd.util.hash_pandas_object(self.target_df, index=False).values)
        h.update(",".join(self.target_cols).encode())
        h.update(str(atom_messages).encode())
        h.update(str(keep_hs).encode())
        h.update(str(sanitize).encode())
        h.update(",".join(self.input_type).encode())
        self.cache_hash = h.hexdigest()

        self._proc_dir = Path(root) / "processed" / self.cache_hash
        if force_reload and self._proc_dir.exists():
            shutil.rmtree(self._proc_dir, ignore_errors=True)
        os.makedirs(self._proc_dir, exist_ok=True)
        super().__init__(root, transform, pre_transform)

        # load everything
        all_pairs = torch.load(self.processed_paths[0], weights_only=False)

        # now prune out any pair whose target contains your sentinel
        if prune_value is not None:
            pruned = []
            for g1, g2 in all_pairs:
                # if *none* of the entries in g1.y equals prune_value, we keep it
                if not (g1.y == prune_value).any():
                    pruned.append((g1, g2))
            self.pairs = pruned
        else:
            self.pairs = all_pairs

    

        self.raw_y = torch.stack([p[0].y.clone() for p in self.pairs])
        self.mean = self.std = None

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
        # Read the SDF file
        data_pairs: list[tuple[Data, Data]] = []

        for sdf_path in self.sdf_paths:
            suppl = Chem.SDMolSupplier(
                str(sdf_path),
                removeHs=not self.keep_hs,
                sanitize=self.sanitize,
            )

            # group mols by reaction tag
            buckets: dict[str, list[Chem.Mol]] = {}
            for mol in suppl:
                if mol is None:
                    continue
                if not mol.HasProp("type") or mol.GetProp("type") not in self.input_type:
                    continue
                if not mol.HasProp("reaction"):
                    warnings.warn(f"{sdf_path.name}: molecule missing 'reaction' tag")
                    continue
                buckets.setdefault(mol.GetProp("reaction"), []).append(mol)

            for rxn, mols in buckets.items():
                if len(mols) != 2:
                    warnings.warn(f"{rxn}: expected exactly 2 mols, got {len(mols)} – skipped")
                    continue

                # featurise
                g1 = featurize_to_Data(mols[0], self.atom_messages)
                g2 = featurize_to_Data(mols[1], self.atom_messages)

                # attach targets
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
                        # Apply the global featurizer to both graphs
                        smiles1 = mols[0].GetProp("smiles") if mols[0].HasProp("smiles") else Chem.MolToSmiles(mols[0])
                        smiles2 = mols[1].GetProp("smiles") if mols[1].HasProp("smiles") else Chem.MolToSmiles(mols[1])
                        gf1 = self.global_featurizer(smiles1)
                        gf2 = self.global_featurizer(smiles2)

                        g1.global_features = torch.tensor(gf1, dtype=torch.float32).view(1, -1)
                        g2.global_features = torch.tensor(gf2, dtype=torch.float32).view(1, -1)
                    except Exception as e:
                        warnings.warn(f"Failed to featurize global features for {rxn}: {e}")
                        continue
                # Add name and reaction type to g1 and g2
                g1.name = mols[0].GetProp("type") if mols[0].HasProp("type") else f"{rxn}_mol1"
                g2.name = mols[1].GetProp("type") if mols[1].HasProp("type") else f"{rxn}_mol2"
                g1.reaction = g2.reaction = rxn
                data_pairs.append((g1, g2))


        if len(data_pairs) == 0:
            raise RuntimeError("No valid (mol1, mol2, target) pairs found! The issue may be in the SDF file or the target DataFrame.")

        torch.save(data_pairs, self.processed_paths[0])


    # -------- statistics & scaling used by your training code ------
    def compute_normalization(self, indices: list[int]):
        """
        Fits μ/σ on the continuous columns *and* computes
        circular means for each periodic column.
        """
        # stack raw y’s for each selected pair
        ys = torch.stack([ self.pairs[i][0].y for i in indices ])  # shape [N, M]
        # continuous columns
        cont = ys[:, self.cont_idx]                               # [N, C]
        self.cont_mean = cont.mean(dim=0) if cont.numel() > 0 else torch.tensor([], dtype=torch.float32)
        self.cont_std  = cont.std(dim=0, unbiased=False).clamp(min=1e-12) if cont.numel() > 0 else torch.tensor([], dtype=torch.float32)

        # periodic columns: convert degrees→radians
        per_deg = ys[:, self.per_idx]                             # [N, P]
        per_rad = per_deg * (math.pi/180)
        # circular mean per column
        sin_sum = per_rad.sin().mean(dim=0)
        cos_sum = per_rad.cos().mean(dim=0)
        self.per_mean_dir = torch.atan2(sin_sum, cos_sum)         # [P]
        
        self.mean = self.cont_mean
        self.std  = self.cont_std
        return self.cont_mean, self.cont_std, self.per_mean_dir

    def apply_normalization(self):
        if self.cont_mean is None or self.per_mean_dir is None:
            raise RuntimeError("Must call compute_normalization first.")



        for g1, g2 in self.pairs:
            y = g1.y.clone().view(-1)   # raw cont + periodic (in degrees)
            # 1) normalize continuous
            y[self.cont_idx] = (y[self.cont_idx] - self.cont_mean) / self.cont_std

            # 2) recenter periodic: deg→rad, shift, wrap to [−π,π], back to rad
            pd = y[self.per_idx] * (math.pi/180)                                    # [P]→rad
            shifted = ((pd - self.per_mean_dir) + math.pi) % (2*math.pi) - math.pi
            # now convert to sin/cos and splice into y
            sinv = shifted.sin()
            cosv = shifted.cos()
            # replace those P columns by 2P columns in the order sin,cos, sin,cos…
            new_per = torch.stack([sinv, cosv], dim=1).view(-1)                    # [2P]
            # finally rebuild y: cont (C) followed by these 2P values
            y = torch.cat([y[self.cont_idx], new_per], dim=0)

            g1.y = g2.y = y.unsqueeze(0)




    # -------------- convenience feature-dimension accessors --------
    @property
    def num_atom_features(self) -> int:
        return self.pairs[0][0].x.size(1)

    @property
    def num_bond_features(self) -> int:
        return self.pairs[0][0].edge_attr.size(1)

    @property
    def num_targets(self) -> int:
        return self.pairs[0][0].y.numel()

    # # @property
    # def get_global_feature_dim(self):
    #     gf = self.pairs[0][0].global_features
    #     if isinstance(gf, np.ndarray):
    #         return gf.size
    #     elif isinstance(gf, torch.Tensor):
    #         return gf.numel()
    #     else:
    #         raise TypeError(f"Unsupported type for global_features: {type(gf)}")
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
