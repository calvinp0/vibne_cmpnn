import logging
from typing import Union

import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

logger = logging.getLogger(__name__)


class CompositeGlobalFeaturizer:
    """
    A composite featurizer that merges outputs from multiple global featurizers
    into a single feature vector.

    Each featurizer must implement a method:
        featurize(mol: Chem.Mol) -> np.ndarray
    """

    def __init__(self, featurizers: list):
        """
        :param featurizers: A list of objects, each with a .featurize(mol: Chem.Mol) method.
        """
        self.featurizers = featurizers

    def featurize(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        feat_list = []
        for featurizer in self.featurizers:
            feats = featurizer.featurize(mol)
            feat_list.append(feats)
        return np.concatenate(feat_list)

    def __call__(self, smiles: str) -> np.ndarray:
        return self.featurize(smiles)


# Morgan Fingerprint Featurizer (binary version)
class MorganBinaryFeaturizer:
    def __init__(self, radius: int = 2, length: int = 2048, includeChirality: bool = True,
                 useCountSimulation: bool = True):
        if radius < 0:
            raise ValueError(f"arg 'radius' must be >= 0! got: {radius}")
        self.radius = radius
        self.length = length
        self.useCountSimulation = useCountSimulation
        self.F = GetMorganGenerator(radius=radius, countSimulation=useCountSimulation,
                                    includeChirality=includeChirality, fpSize=length)

    def featurize(self, mol: Chem.Mol) -> np.ndarray:
        if self.useCountSimulation:
            # When count simulation is used, you might get counts—here we convert counts to binary.
            fp = self.F.GetCountFingerprintAsNumPy(mol)
            # Convert counts to binary (i.e. 0 if 0, else 1):
            return (fp > 0).astype(int)
        else:
            # When count simulation is off, use the bit-vector fingerprint directly:
            fp = self.F.GetFingerprint(mol)
            # Convert to a NumPy array (this should match AllChem.GetMorganFingerprintAsBitVect behavior)
            arr = np.array(fp)
            return arr


# RDKit 2D Featurizer (standard descriptors)
class RDKit2DFeaturizer:
    def __init__(self):
        logger.warning(
            "The RDKit 2D features can deviate significantly from a normal distribution. "
            "Consider manually scaling them."
        )

    def featurize(self, mol: Chem.Mol) -> np.ndarray:
        features = np.array([func(mol) for name, func in Descriptors.descList], dtype=float)
        return features


# V1RDKit2DNormalizedFeaturizer
class RDKit2DNormalizedFeaturizer:
    def __init__(self):
        self.generator = rdNormalizedDescriptors.RDKit2DNormalized()

    def featurize(self, mol: Chem.Mol) -> np.ndarray:
        smiles = Chem.MolToSmiles(mol)
        results = self.generator.process(smiles)
        processed, features = results[0], results[1:]
        if not processed:
            raise ValueError(f"Failed to process SMILES: {smiles}")
        return np.array(features, dtype=float)


class ChargeFeaturizer:
    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol}")
        return np.array([Chem.GetFormalCharge(mol)], dtype=float)

    def featurize(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        return self.__call__(mol)

    def __len__(self):
        return 1
