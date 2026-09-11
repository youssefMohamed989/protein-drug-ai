"""
Advanced 3D molecular visualization of representative toxicophore compounds.
Selects real high-SHAP-relevant compounds from the dataset (high LogP / high aromatic
ring count) for SR-MMP and NR-AhR, embeds 3D conformers via RDKit (ETKDG), and renders
publication-quality 3D stick/ball molecular structures highlighting the toxicophore-relevant
regions (lipophilic core for SR-MMP; planar aromatic system for NR-AhR).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

plt.rcParams.update({'font.size': 9})

df = pd.read_csv('/home/claude/project/data/tox21.csv')

ATOM_COLORS = {
    'C': '#4C4C4C', 'N': '#3050F8', 'O': '#FF0D0D', 'S': '#FFC832',
    'Cl': '#1FF01F', 'F': '#90E050', 'Br': '#A62929', 'H': '#DDDDDD', 'P': '#FF8000'
}
ATOM_RADII = {
    'C': 70, 'N': 65, 'O': 60, 'S': 100, 'Cl': 100, 'F': 50, 'Br': 115, 'H': 25, 'P': 100
}

def embed_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def plot_molecule_3d(ax, mol, title, highlight_type=None):
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if symbols[i] == 'H' or symbols[j] == 'H':
            continue  # skip explicit H bonds for clarity
        x = [coords[i][0], coords[j][0]]
        y = [coords[i][1], coords[j][1]]
        z = [coords[i][2], coords[j][2]]
        ax.plot(x, y, z, color='#888888', linewidth=1.8, zorder=1)

    # highlight aromatic ring atoms (planarity, for AhR) or heavy lipophilic carbon skeleton (SR-MMP)
    ri = mol.GetRingInfo()
    aromatic_atoms = set()
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            aromatic_atoms.update(ring)

    for idx, (sym, pos) in enumerate(zip(symbols, coords)):
        if sym == 'H':
            continue
        color = ATOM_COLORS.get(sym, '#FF69B4')
        size = ATOM_RADII.get(sym, 70)
        edge = 'none'
        lw = 0
        if highlight_type == 'aromatic' and idx in aromatic_atoms:
            edge = '#E63946'; lw = 1.8; size *= 1.35
        elif highlight_type == 'lipophilic' and sym == 'C' and idx not in aromatic_atoms:
            edge = '#F4A261'; lw = 1.4; size *= 1.2
        ax.scatter(*pos, color=color, s=size, edgecolor=edge, linewidth=lw, depthshade=True, zorder=2)

    ax.set_title(title, fontweight='bold', fontsize=9.5)
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none'); ax.zaxis.pane.set_edgecolor('none')

# --- Select real representative compounds from dataset by descriptor extremes ---
def compute_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Descriptors.MolLogP(mol), Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol), mol.GetNumHeavyAtoms()

props = []
for smi in df['smiles']:
    p = compute_props(smi)
    if p is not None and p[3] >= 12 and p[3] <= 40:  # moderate size for clear visualization
        props.append((smi, *p))
props_df = pd.DataFrame(props, columns=['smiles','LogP','ArRings','Fsp3','HeavyAtoms'])

# High-LogP, low-aromatic-ring compound -> SR-MMP-relevant lipophilic toxicophore example
high_logp_pool = props_df[(props_df['LogP'] > 4) & (props_df['LogP'] < 7) & (props_df['HeavyAtoms'] < 30)]
high_logp = high_logp_pool.sort_values('LogP', ascending=False).iloc[2] if len(high_logp_pool) > 2 else props_df.sort_values('LogP', ascending=False).iloc[3]
# High aromatic-ring-count, low-Fsp3 compound -> NR-AhR-relevant planar aromatic toxicophore example
high_aromatic = props_df[(props_df['ArRings'] >= 3)].sort_values('Fsp3').iloc[2]
# Low-risk comparator: low LogP, high Fsp3, few aromatic rings
low_risk_pool = props_df[(props_df['ArRings'] <= 1) & (props_df['Fsp3'] >= 0.6) & (props_df['LogP'] < 1.5)]
low_risk = low_risk_pool.sort_values('LogP').iloc[len(low_risk_pool)//2] if len(low_risk_pool) > 0 else props_df.sort_values('LogP').iloc[5]

print("High LogP (SR-MMP-relevant):", high_logp['smiles'], "LogP=", round(high_logp['LogP'],2))
print("High aromatic (NR-AhR-relevant):", high_aromatic['smiles'], "ArRings=", int(high_aromatic['ArRings']), "Fsp3=", round(high_aromatic['Fsp3'],2))
print("Low-risk comparator:", low_risk['smiles'], "LogP=", round(low_risk['LogP'],2), "Fsp3=", round(low_risk['Fsp3'],2))

fig = plt.figure(figsize=(15, 5.5))

mols_info = [
    (high_logp['smiles'], f"A. High-lipophilicity toxicophore\n(SR-MMP risk)  LogP={high_logp['LogP']:.2f}", 'lipophilic'),
    (high_aromatic['smiles'], f"B. Planar polyaromatic toxicophore\n(NR-AhR risk)  Aromatic rings={int(high_aromatic['ArRings'])}, Fsp3={high_aromatic['Fsp3']:.2f}", 'aromatic'),
    (low_risk['smiles'], f"C. Low-risk comparator\n(saturated, low LogP)  LogP={low_risk['LogP']:.2f}, Fsp3={low_risk['Fsp3']:.2f}", None),
]

for i, (smi, title, htype) in enumerate(mols_info):
    mol3d = embed_3d(smi)
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    if mol3d is not None:
        plot_molecule_3d(ax, mol3d, title, highlight_type=htype)
    else:
        ax.set_title(title + '\n(embedding failed)', fontsize=9)
    ax.view_init(elev=18, azim=35)

legend_elems = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#4C4C4C', markersize=9, label='Carbon'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#3050F8', markersize=9, label='Nitrogen'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF0D0D', markersize=9, label='Oxygen'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#FFC832', markersize=9, label='Sulfur'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#90E050', markersize=9, label='Halogen'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#E63946', markeredgewidth=2, markersize=9, label='Aromatic-ring atom (AhR toxicophore)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#F4A261', markeredgewidth=2, markersize=9, label='Lipophilic aliphatic carbon (SR-MMP toxicophore)'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
fig.suptitle('3D molecular conformations of representative toxicophore-bearing compounds (MMFF-optimized)', fontweight='bold', fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig8_3d_toxicophore.png', dpi=300, bbox_inches='tight')
print("Saved fig8_3d_toxicophore.png")
