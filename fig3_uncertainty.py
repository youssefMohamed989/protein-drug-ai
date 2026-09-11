import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 10})

with open('/home/claude/project/models/uncertainty_analysis.pkl', 'rb') as f:
    ua = pickle.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel A: Reliability diagram ---
ax = axes[0]
conf = np.array(ua['calib_conf']); acc = np.array(ua['calib_acc']); count = np.array(ua['calib_count'])
valid = ~np.isnan(conf)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Perfect calibration')
sizes = 200 * count[valid] / count[valid].max()
sc = ax.scatter(conf[valid], acc[valid], s=sizes, c='#8172B2', edgecolor='black', linewidth=1, alpha=0.85, zorder=3)
ax.bar(conf[valid], acc[valid], width=0.08, alpha=0.25, color='#8172B2')
ax.set_xlabel('Mean predicted probability (confidence)')
ax.set_ylabel('Observed frequency (accuracy)')
ax.set_title(f"A. Reliability diagram (pooled, 12 endpoints)\nECE = {ua['ece']:.4f}  |  Brier = {ua['brier']:.4f}", fontweight='bold', fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# --- Panel B: Selective prediction curve ---
ax2 = axes[1]
frac = np.array(ua['reject_fractions']) * 100
sauc = np.array(ua['selective_auc'])
ax2.plot(frac, sauc, marker='o', color='#C44E52', linewidth=2, markersize=6)
ax2.axhline(sauc[0], color='gray', linestyle=':', linewidth=1, label='No rejection (baseline)')
ax2.set_xlabel('Fraction of highest-uncertainty predictions rejected (%)')
ax2.set_ylabel('Retained-set ROC-AUC')
ax2.set_title('B. Selective prediction: naive uncertainty\nrejection under class imbalance', fontweight='bold', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.annotate('AUC declines: rejected "low-uncertainty"\nset is dominated by trivial negatives',
             xy=(30, sauc[6]), xytext=(15, 0.62),
             arrowprops=dict(arrowstyle='->', color='black', lw=1),
             fontsize=7.5, ha='center')

# --- Panel C: per-task low- vs high-uncertainty AUC split ---
ax3 = axes[2]
tasks = list(ua['per_task_stats'].keys())
low_auc = [ua['per_task_stats'][t]['auc_low_unc'] for t in tasks]
high_auc = [ua['per_task_stats'][t]['auc_high_unc'] for t in tasks]
x = np.arange(len(tasks))
w = 0.35
ax3.barh(x - w/2, low_auc, w, label='Below-median epistemic std', color='#4C72B0', edgecolor='black', linewidth=0.4)
ax3.barh(x + w/2, high_auc, w, label='Above-median epistemic std', color='#DD8452', edgecolor='black', linewidth=0.4)
ax3.set_yticks(x)
ax3.set_yticklabels(tasks, fontsize=8)
ax3.set_xlabel('ROC-AUC')
ax3.set_title('C. Discriminative power by\nensemble-disagreement stratum', fontweight='bold', fontsize=10)
ax3.legend(fontsize=7, loc='lower right')
ax3.axvline(0.5, color='gray', linestyle=':', linewidth=1)
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig3_uncertainty.png', dpi=300, bbox_inches='tight')
print("Saved fig3_uncertainty.png")
