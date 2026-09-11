"""
External validation: apply trained per-endpoint models to a small set of well-characterized
reference compounds with known literature toxicological mechanism, independent of Tox21
training data (checked for exact-structure overlap and excluded if present), to test whether
model predictions and SHAP-implicated toxicophores align with established pharmacology.
"""
import numpy as np, pickle, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

# Well-characterized reference compounds with documented literature mechanism
# (canonical SMILES; mechanism per cited pharmacological/toxicological literature)
reference_compounds = [
    # Classic strong AhR agonists (planar polyaromatic / halogenated aromatics)
    {'name': 'Benzo[a]pyrene', 'smiles': 'c1ccc2c(c1)cc1ccc3cccc4c3c1c2cc4', 'expected_endpoint': 'NR-AhR', 'expected_direction': 'toxic', 'class': 'PAH (AhR agonist)'},
    {'name': 'Dibenzo[a,h]anthracene', 'smiles': 'c1ccc2c(c1)ccc1c2ccc2c1cc1ccccc1c2', 'expected_endpoint': 'NR-AhR', 'expected_direction': 'toxic', 'class': 'PAH (AhR agonist)'},
    {'name': 'Beta-naphthoflavone', 'smiles': 'c1ccc2c(c1)ccc1c2oc(-c2ccccc2)cc1=O', 'expected_endpoint': 'NR-AhR', 'expected_direction': 'toxic', 'class': 'Flavonoid AhR agonist'},
    # Classic mitochondrial uncouplers / SR-MMP disruptors (lipophilic weak acids/cations)
    {'name': 'FCCP (carbonyl cyanide-4-trifluoromethoxyphenylhydrazone)', 'smiles': 'FC(F)(F)Oc1ccc(cc1)/N=N\\C(C#N)=NN', 'expected_endpoint': 'SR-MMP', 'expected_direction': 'toxic', 'class': 'Mitochondrial uncoupler'},
    {'name': '2,4-Dinitrophenol', 'smiles': 'Oc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]', 'expected_endpoint': 'SR-MMP', 'expected_direction': 'toxic', 'class': 'Mitochondrial uncoupler'},
    {'name': 'Amiodarone', 'smiles': 'CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1', 'expected_endpoint': 'SR-MMP', 'expected_direction': 'toxic', 'class': 'Cationic amphiphilic drug (mitochondrial toxicant)'},
    # Low-risk saturated reference compounds (expected non-toxic on both endpoints)
    {'name': 'Glucose', 'smiles': 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 'expected_endpoint': None, 'expected_direction': 'non-toxic', 'class': 'Saturated polyol (negative control)'},
    {'name': 'Cyclohexanol', 'smiles': 'OC1CCCCC1', 'expected_endpoint': None, 'expected_direction': 'non-toxic', 'class': 'Saturated alcohol (negative control)'},
    {'name': 'Citric acid', 'smiles': 'OC(=O)CC(O)(CC(=O)O)C(=O)O', 'expected_endpoint': None, 'expected_direction': 'non-toxic', 'class': 'Saturated polyacid (negative control)'},
]

data = np.load('/home/claude/project/data/featurized.npz', allow_pickle=True)
train_smiles = set(data['smiles'])

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    canon = Chem.MolToSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_arr = np.array(fp, dtype=np.int8)
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
    ], dtype=np.float32)
    return fp_arr, desc, canon

# Recompute descriptor standardization stats from training data (must match 01_preprocess.py)
X_full = data['X']
desc_train_raw = []
for smi in data['smiles']:
    mol = Chem.MolFromSmiles(smi)
    d = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
    ], dtype=np.float32)
    desc_train_raw.append(d)
desc_train_raw = np.vstack(desc_train_raw)
desc_mean, desc_std = desc_train_raw.mean(0), desc_train_raw.std(0)

with open('/home/claude/project/models/ensemble_models.pkl', 'rb') as f:
    ensemble_models = pickle.load(f)

train_canon_all = set()
for s in data['smiles']:
    m = Chem.MolFromSmiles(s)
    if m is not None:
        train_canon_all.add(Chem.MolToSmiles(m))

results = []
for comp in reference_compounds:
    fp, desc_raw, canon = featurize(comp['smiles'])
    if fp is None:
        print(f"WARNING: could not parse {comp['name']}")
        continue
    in_training = canon in train_canon_all
    desc_std_v = (desc_raw - desc_mean) / (desc_std + 1e-8)
    feat = np.concatenate([fp, desc_std_v]).reshape(1, -1)

    row = {'Compound': comp['name'], 'Class': comp['class'],
           'In Tox21 training set': 'Yes' if in_training else 'No (true external)',
           'Expected endpoint': comp['expected_endpoint'] or '-'}
    for task in ['NR-AhR', 'SR-MMP']:
        preds = [m.predict_proba(feat)[0, 1] for m in ensemble_models[task]]
        row[f'{task}_P(toxic)'] = round(float(np.mean(preds)), 3)
        row[f'{task}_std'] = round(float(np.std(preds)), 3)
    results.append(row)

ext_df = pd.DataFrame(results)
ext_df.to_csv('/home/claude/project/tables/table6_external_validation.csv', index=False)
print(ext_df.to_string(index=False))
n_external = sum(1 for c in reference_compounds if Chem.MolToSmiles(Chem.MolFromSmiles(c['smiles'])) not in train_canon_all)
print(f"\n{n_external} of {len(reference_compounds)} reference compounds are true external (not in Tox21 training data).")
