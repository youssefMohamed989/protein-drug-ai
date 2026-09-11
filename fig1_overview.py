import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.45, wspace=0.35)

# --- Panel A: pipeline schematic ---
ax0 = fig.add_subplot(gs[0, :])
ax0.set_xlim(0, 14); ax0.set_ylim(0, 3); ax0.axis('off')
ax0.set_title('A. End-to-end multi-task uncertainty-aware toxicity prediction pipeline', loc='left', fontweight='bold', fontsize=12)

boxes = [
    (0.3, 1, 2.0, 1, "Tox21\nSMILES\n(n=8,006)", '#4C72B0'),
    (2.7, 1, 2.0, 1, "Featurization\nMorgan-FP(1024)\n+10 descriptors", '#55A868'),
    (5.1, 1, 2.1, 1, "Multi-task split\n70/10/20\n(train/val/test)", '#C44E52'),
    (7.6, 1, 2.1, 1, "Deep ensemble\n(8x MLP)\nper endpoint", '#8172B2'),
    (10.1, 1, 1.9, 1, "Uncertainty\n+ calibration\nanalysis", '#CCB974'),
    (12.35, 1, 1.6, 1, "SHAP\ninterpretation", '#64B5CD'),
]
for x, y, w, h, label, color in boxes:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                          linewidth=1.3, edgecolor='black', facecolor=color, alpha=0.85)
    ax0.add_patch(box)
    ax0.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    y = boxes[i][1] + boxes[i][3]/2
    ax0.annotate('', xy=(x2, y), xytext=(x1, y),
                 arrowprops=dict(arrowstyle='-|>', lw=1.8, color='black'))

# --- Panel B: label prevalence per task ---
ax1 = fig.add_subplot(gs[1, 0])
data = np.load('/home/claude/project/data/featurized.npz')
Y, mask = data['Y'], data['mask']
TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
pos_rate = [Y[mask[:,i], i].mean()*100 for i in range(12)]
colors = plt.cm.viridis(np.linspace(0.15, 0.9, 12))
bars = ax1.barh(TASKS, pos_rate, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Positive rate (%)')
ax1.set_title('B. Class imbalance across endpoints', fontweight='bold', fontsize=11)
ax1.invert_yaxis()
for bar, v in zip(bars, pos_rate):
    ax1.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=8)

# --- Panel C: missingness per task ---
ax2 = fig.add_subplot(gs[1, 1])
miss_rate = [(1 - mask[:,i].mean())*100 for i in range(12)]
bars2 = ax2.barh(TASKS, miss_rate, color='#CC8963', edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Missing labels (%)')
ax2.set_title('C. Label missingness (multi-task masking)', fontweight='bold', fontsize=11)
ax2.invert_yaxis()
ax2.set_yticklabels([])

# --- Panel D: molecular property distribution (MolWt vs LogP density) ---
ax3 = fig.add_subplot(gs[1, 2])
X = data['X']
molwt_std = X[:, 1024]  # standardized MolWt
logp_std = X[:, 1025]   # standardized LogP
h = ax3.hist2d(molwt_std, logp_std, bins=40, cmap='mako' if 'mako' in plt.colormaps() else 'viridis')
ax3.set_xlabel('MolWt (standardized)')
ax3.set_ylabel('LogP (standardized)')
ax3.set_title('D. Chemical space coverage', fontweight='bold', fontsize=11)
plt.colorbar(h[3], ax=ax3, label='Compound density', shrink=0.8)

plt.savefig('/home/claude/project/figures/fig1_overview.png', dpi=300, bbox_inches='tight')
print("Saved fig1_overview.png")
