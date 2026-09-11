import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({'font.size': 9})

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

with open('/home/claude/project/models/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)
results = ens['results']

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()
colors = plt.cm.tab20(np.linspace(0, 1, 12))

for i, task in enumerate(TASKS):
    ax = axes[i]
    r = results[task]
    fpr, tpr, _ = roc_curve(r['y_true'], r['y_pred_mean'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], linewidth=2.2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.15, color=colors[i])
    ax.set_title(task, fontweight='bold', fontsize=10)
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.suptitle('ROC curves for all 12 Tox21 endpoints (deep ensemble, held-out test set)', fontweight='bold', fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig6_roc.png', dpi=300, bbox_inches='tight')
print("Saved fig6_roc.png")
