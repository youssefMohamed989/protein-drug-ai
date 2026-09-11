import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 10})

with open('/home/claude/project/models/shap_results.pkl', 'rb') as f:
    shap_res = pickle.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, task in zip(axes, ['SR-MMP', 'NR-AhR']):
    res = shap_res[task]
    top = res['top_features'][:10]
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    colors = ['#C44E52' if 'FP_' in n else '#4C72B0' for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'{task}\n(top 10 features, XGBoost surrogate)', fontweight='bold', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

legend_elems = [plt.Rectangle((0,0),1,1, color='#4C72B0', label='Physicochemical descriptor'),
                plt.Rectangle((0,0),1,1, color='#C44E52', label='Morgan-FP substructure bit')]
fig.legend(handles=legend_elems, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
fig.suptitle('D. Feature attribution reveals mechanistically plausible toxicophores', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig4_shap.png', dpi=300, bbox_inches='tight')
print("Saved fig4_shap.png")
