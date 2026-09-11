import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 10})

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

with open('/home/claude/project/models/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)
with open('/home/claude/project/models/baseline_results.pkl', 'rb') as f:
    baselines = pickle.load(f)

ensemble_auc = [ens['results'][t]['auc'] for t in TASKS]
rf_auc = [baselines['RandomForest'][t] for t in TASKS]
xgb_auc = [baselines['XGBoost'][t] for t in TASKS]
mlp_auc = [baselines['Single_MLP'][t] for t in TASKS]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

# --- Panel A: grouped bar chart, per-task AUC across 4 models ---
ax = axes[0]
x = np.arange(len(TASKS))
width = 0.2
ax.bar(x - 1.5*width, rf_auc, width, label='Random Forest', color='#4C72B0', edgecolor='black', linewidth=0.4)
ax.bar(x - 0.5*width, xgb_auc, width, label='XGBoost', color='#DD8452', edgecolor='black', linewidth=0.4)
ax.bar(x + 0.5*width, mlp_auc, width, label='Single MLP', color='#55A868', edgecolor='black', linewidth=0.4)
ax.bar(x + 1.5*width, ensemble_auc, width, label='Deep Ensemble (ours)', color='#8172B2', edgecolor='black', linewidth=0.4)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Random baseline')
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Test ROC-AUC')
ax.set_ylim(0.4, 1.0)
ax.set_title('A. Per-endpoint ROC-AUC across four modeling approaches', fontweight='bold')
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.3)

# --- Panel B: mean AUC summary with error bars (std across 12 tasks) ---
ax2 = axes[1]
means = [np.mean(rf_auc), np.mean(xgb_auc), np.mean(mlp_auc), np.mean(ensemble_auc)]
stds = [np.std(rf_auc), np.std(xgb_auc), np.std(mlp_auc), np.std(ensemble_auc)]
labels = ['Random\nForest', 'XGBoost', 'Single\nMLP', 'Deep\nEnsemble']
colors = ['#4C72B0', '#DD8452', '#55A868', '#8172B2']
bars = ax2.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor='black', linewidth=0.8)
for bar, m in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, m + 0.02, f'{m:.3f}', ha='center', fontweight='bold', fontsize=9)
ax2.set_ylabel('Mean ROC-AUC (± SD across 12 endpoints)')
ax2.set_ylim(0.4, 1.0)
ax2.set_title('B. Overall model comparison', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig2_performance.png', dpi=300, bbox_inches='tight')
print("Saved fig2_performance.png")
