import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 10})

with open('/home/claude/project/models/shap_results.pkl', 'rb') as f:
    shap_res = pickle.load(f)

DESC_NAMES = ['MolWt','LogP','NumHDonors','NumHAcceptors','TPSA',
              'NumRotatableBonds','NumAromaticRings','NumRings','FractionCSP3','HeavyAtomCount']
feature_names = [f'FP_{i}' for i in range(1024)] + DESC_NAMES

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

panels = [
    ('SR-MMP', 'LogP', axes[0,0], 'A'),
    ('SR-MMP', 'FractionCSP3', axes[0,1], 'B'),
    ('NR-AhR', 'FractionCSP3', axes[1,0], 'C'),
    ('NR-AhR', 'NumAromaticRings', axes[1,1], 'D'),
]

for task, feat, ax, panel_label in panels:
    res = shap_res[task]
    feat_idx = feature_names.index(feat)
    X_sample = res['X_sample']
    shap_values = res['shap_values']
    x_vals = X_sample[:, feat_idx]
    y_vals = shap_values[:, feat_idx]
    sc = ax.scatter(x_vals, y_vals, c=y_vals, cmap='coolwarm', s=25, edgecolor='black', linewidth=0.3, alpha=0.8)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel(f'{feat} (standardized)')
    ax.set_ylabel('SHAP value (impact on toxicity logit)')
    ax.set_title(f'{panel_label}. {task}: {feat} dependence', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.25)
    z = np.polyfit(x_vals, y_vals, 2)
    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color='black', linewidth=2, linestyle='--', alpha=0.7, label='Local trend (deg-2 fit)')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig5_dependence.png', dpi=300, bbox_inches='tight')
print("Saved fig5_dependence.png")
