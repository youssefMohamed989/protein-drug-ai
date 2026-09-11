"""
Tox21 preprocessing and molecular featurization.
Generates Morgan fingerprints + physicochemical descriptors for each compound.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

np.random.seed(42)

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

df = pd.read_csv('/home/claude/project/data/tox21.csv')
print(f"Raw compounds: {len(df)}")

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_arr = np.array(fp, dtype=np.int8)
    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
    ], dtype=np.float32)
    return fp_arr, desc

fps, descs, valid_idx = [], [], []
for i, smi in enumerate(df['smiles']):
    result = featurize(smi)
    if result is not None:
        fps.append(result[0])
        descs.append(result[1])
        valid_idx.append(i)

df_valid = df.iloc[valid_idx].reset_index(drop=True)
X_fp = np.vstack(fps)
X_desc = np.vstack(descs)

# standardize descriptors
X_desc = (X_desc - X_desc.mean(0)) / (X_desc.std(0) + 1e-8)
X = np.hstack([X_fp, X_desc])

Y = df_valid[TASKS].values.astype(np.float32)  # NaN = missing label
mask = ~np.isnan(Y)
Y_filled = np.nan_to_num(Y, nan=0.0)

print(f"Valid compounds after RDKit parsing: {len(df_valid)}")
print(f"Feature matrix: {X.shape}, Label matrix: {Y.shape}")
print(f"Label missingness per task:\n{pd.DataFrame(mask, columns=TASKS).mean().apply(lambda x: f'{100*(1-x):.1f}% missing')}")

np.savez('/home/claude/project/data/featurized.npz', X=X, Y=Y_filled, mask=mask,
         smiles=df_valid['smiles'].values, mol_id=df_valid['mol_id'].values)
print("Saved featurized.npz")
print(f"Positive rate per task:\n{pd.DataFrame(np.where(mask, Y_filled, np.nan), columns=TASKS).mean()}")
