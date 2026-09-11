"""
Expanded external validation across 5 mechanistically distinct endpoints (NR-AhR, SR-MMP,
NR-ER, NR-AR, SR-ARE, SR-p53) using 20 well-characterized literature reference compounds,
combined with a formal applicability domain (AD) analysis: for each external compound we
compute the Tanimoto similarity to its nearest neighbor in the Tox21 training set (Morgan
fingerprints), which is the standard QSAR approach for assessing whether a prediction falls
inside or outside the model's domain of reliable applicability.
"""
import numpy as np, pickle, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

reference_compounds = [
    # --- NR-AhR: canonical planar (halo)aromatic agonists ---
    {'name': 'Benzo[a]pyrene', 'smiles': 'c1ccc2c(c1)cc1ccc3cccc4c3c1c2cc4', 'endpoint': 'NR-AhR', 'expected': 'toxic', 'class': 'PAH (AhR agonist)'},
    {'name': 'Dibenzo[a,h]anthracene', 'smiles': 'c1ccc2c(c1)ccc1c2ccc2c1cc1ccccc1c2', 'endpoint': 'NR-AhR', 'expected': 'toxic', 'class': 'PAH (AhR agonist)'},
    {'name': 'Beta-naphthoflavone', 'smiles': 'c1ccc2c(c1)ccc1c2oc(-c2ccccc2)cc1=O', 'endpoint': 'NR-AhR', 'expected': 'toxic', 'class': 'Flavonoid AhR agonist'},
    {'name': '3-Methylcholanthrene', 'smiles': 'CC1=C2CCC3=C2C(=CC4=C3C=CC5=CC=CC=C54)C=C1', 'endpoint': 'NR-AhR', 'expected': 'toxic', 'class': 'PAH (potent AhR agonist)'},
    # --- SR-MMP: mitochondrial membrane potential disruptors ---
    {'name': 'FCCP', 'smiles': 'FC(F)(F)Oc1ccc(cc1)/N=N\\C(C#N)=NN', 'endpoint': 'SR-MMP', 'expected': 'toxic', 'class': 'Mitochondrial uncoupler'},
    {'name': '2,4-Dinitrophenol', 'smiles': 'Oc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]', 'endpoint': 'SR-MMP', 'expected': 'toxic', 'class': 'Mitochondrial uncoupler'},
    {'name': 'Amiodarone', 'smiles': 'CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1', 'endpoint': 'SR-MMP', 'expected': 'toxic', 'class': 'Cationic amphiphilic drug'},
    {'name': 'Valinomycin', 'smiles': 'CC(C)C1C(=O)OC(C)C(=O)OC(C(C)C)C(=O)OC(C)C(=O)OC(C(C)C)C(=O)OC(C)C(=O)OC(C(C)C)C(=O)OC(C)C(=O)OC(C(C)C)C(=O)O1', 'endpoint': 'SR-MMP', 'expected': 'toxic', 'class': 'K+ ionophore (uncoupler)'},
    # --- NR-ER: estrogen receptor agonists ---
    {'name': '17-beta-Estradiol', 'smiles': 'C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O', 'endpoint': 'NR-ER', 'expected': 'toxic', 'class': 'Endogenous estrogen (ER agonist)'},
    {'name': 'Diethylstilbestrol', 'smiles': 'CC/C(=C(\\CC)c1ccc(O)cc1)c1ccc(O)cc1', 'endpoint': 'NR-ER', 'expected': 'toxic', 'class': 'Synthetic estrogen (potent ER agonist)'},
    {'name': 'Bisphenol A', 'smiles': 'CC(C)(c1ccc(O)cc1)c1ccc(O)cc1', 'endpoint': 'NR-ER', 'expected': 'toxic', 'class': 'Xenoestrogen (weak ER agonist)'},
    # --- NR-AR: androgen receptor agonists ---
    {'name': 'Testosterone', 'smiles': 'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C', 'endpoint': 'NR-AR', 'expected': 'toxic', 'class': 'Endogenous androgen (AR agonist)'},
    {'name': '5-alpha-Dihydrotestosterone', 'smiles': 'CC12CCC3C(C1CCC2O)CCC4C3(CCC(=O)C4)C', 'endpoint': 'NR-AR', 'expected': 'toxic', 'class': 'Potent endogenous androgen'},
    # --- SR-ARE: Nrf2/antioxidant response element activators ---
    {'name': 'Tert-butylhydroquinone', 'smiles': 'CC(C)(C)c1cc(O)ccc1O', 'endpoint': 'SR-ARE', 'expected': 'toxic', 'class': 'Nrf2/ARE activator (phenolic antioxidant)'},
    {'name': 'Sulforaphane', 'smiles': 'CS(=O)CCCCN=C=S', 'endpoint': 'SR-ARE', 'expected': 'toxic', 'class': 'Isothiocyanate (Nrf2/ARE activator)'},
    # --- SR-p53: genotoxicants / DNA-damage inducers ---
    {'name': 'Methyl methanesulfonate', 'smiles': 'COS(=O)(=O)C', 'endpoint': 'SR-p53', 'expected': 'toxic', 'class': 'Alkylating genotoxicant'},
    {'name': 'Etoposide', 'smiles': 'COc1cc(cc(OC)c1O)[C@@H]1c2cc3c(cc2[C@@H](O[C@@H]2O[C@@H]4CO[C@H](C)O[C@H]4[C@H](O)[C@H]2O)[C@@H]2COC(=O)[C@@H]12)OCO3', 'endpoint': 'SR-p53', 'expected': 'toxic', 'class': 'Topoisomerase II inhibitor (genotoxicant)'},
    # --- Negative controls: saturated, non-reactive compounds, no expected liability ---
    {'name': 'Glucose', 'smiles': 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O', 'endpoint': None, 'expected': 'non-toxic', 'class': 'Saturated polyol (negative control)'},
    {'name': 'Cyclohexanol', 'smiles': 'OC1CCCCC1', 'endpoint': None, 'expected': 'non-toxic', 'class': 'Saturated alcohol (negative control)'},
    {'name': 'Citric acid', 'smiles': 'OC(=O)CC(O)(CC(=O)O)C(=O)O', 'endpoint': None, 'expected': 'non-toxic', 'class': 'Saturated polyacid (negative control)'},
]

print(f"Total reference compounds: {len(reference_compounds)}")

data = np.load('/home/claude/project/data/featurized.npz', allow_pickle=True)
train_smiles_raw = data['smiles']

# Build training fingerprints (for AD/Tanimoto) and canonical smiles set (for overlap check)
train_canon_all = set()
train_fps = []
for s in train_smiles_raw:
    m = Chem.MolFromSmiles(s)
    if m is not None:
        train_canon_all.add(Chem.MolToSmiles(m))
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024))

# descriptor standardization stats (must match training featurization)
desc_train_raw = []
for smi in train_smiles_raw:
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

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None
    canon = Chem.MolToSmiles(mol)
    fp_bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_arr = np.array(fp_bv, dtype=np.int8)
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
    ], dtype=np.float32)
    return fp_arr, desc, canon, fp_bv

with open('/home/claude/project/models/ensemble_models.pkl', 'rb') as f:
    ensemble_models = pickle.load(f)

results = []
failed = []
for comp in reference_compounds:
    fp, desc_raw, canon, fp_bv = featurize(comp['smiles'])
    if fp is None:
        failed.append(comp['name'])
        print(f"WARNING: could not parse {comp['name']}")
        continue
    in_training = canon in train_canon_all

    # Applicability domain: max Tanimoto similarity to any training-set compound
    sims = DataStructs.BulkTanimotoSimilarity(fp_bv, train_fps)
    max_sim = float(np.max(sims))

    desc_std_v = (desc_raw - desc_mean) / (desc_std + 1e-8)
    feat = np.concatenate([fp, desc_std_v]).reshape(1, -1)

    endpoint = comp['endpoint']
    if endpoint is not None:
        preds = [m.predict_proba(feat)[0, 1] for m in ensemble_models[endpoint]]
        p_mean, p_std = float(np.mean(preds)), float(np.std(preds))
    else:
        # negative controls: report mean across all 12 endpoints
        all_preds = []
        for t in TASKS:
            pr = [m.predict_proba(feat)[0, 1] for m in ensemble_models[t]]
            all_preds.append(np.mean(pr))
        p_mean, p_std = float(np.mean(all_preds)), float(np.std(all_preds))
        endpoint = 'mean of 12 endpoints'

    predicted_class = 'toxic' if p_mean >= 0.5 else 'non-toxic'
    correct = (predicted_class == comp['expected'])

    results.append({
        'Compound': comp['name'], 'Class': comp['class'], 'Endpoint': endpoint,
        'In training set': 'Yes' if in_training else 'No (external)',
        'Max Tanimoto to training set': round(max_sim, 3),
        'AD status': 'Inside AD (sim>=0.4)' if max_sim >= 0.4 else 'Outside AD (sim<0.4)',
        'Expected': comp['expected'], 'Predicted P(toxic)': round(p_mean, 3),
        'Ensemble SD': round(p_std, 3), 'Predicted class': predicted_class,
        'Correct': 'Yes' if correct else 'No',
    })

ext_df = pd.DataFrame(results)
ext_df.to_csv('/home/claude/project/tables/table6_external_validation.csv', index=False)
pd.set_option('display.width', 200)
print(ext_df.to_string(index=False))

n_external = (ext_df['In training set'] == 'No (external)').sum()
n_correct_external = ((ext_df['In training set'] == 'No (external)') & (ext_df['Correct'] == 'Yes')).sum()
n_correct_all = (ext_df['Correct'] == 'Yes').sum()
print(f"\n{n_external} of {len(ext_df)} compounds are true external (absent from Tox21 training data).")
print(f"External accuracy: {n_correct_external}/{n_external} = {100*n_correct_external/max(n_external,1):.1f}%")
print(f"Overall accuracy (all 20): {n_correct_all}/{len(ext_df)} = {100*n_correct_all/len(ext_df):.1f}%")

# AD-stratified accuracy: inside vs outside applicability domain
inside = ext_df[ext_df['AD status'].str.startswith('Inside')]
outside = ext_df[ext_df['AD status'].str.startswith('Outside')]
print(f"\nInside AD (Tanimoto>=0.4): n={len(inside)}, accuracy={100*(inside['Correct']=='Yes').mean():.1f}%")
print(f"Outside AD (Tanimoto<0.4): n={len(outside)}, accuracy={100*(outside['Correct']=='Yes').mean():.1f}%" if len(outside) else "Outside AD: n=0")
