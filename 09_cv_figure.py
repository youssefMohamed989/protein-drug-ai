import numpy as np, pickle, pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

with open('/home/claude/project/models/cv_results.pkl', 'rb') as f:
    cvd = pickle.load(f)
summary = cvd['summary']

# Table 7: CV summary
rows = []
for t in TASKS:
    row = {'Endpoint': t}
    for model in ['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']:
        s = summary[t][model]
        row[f'{model} (mean AUC)'] = round(s['mean'], 3)
        row[f'{model} (95% CI ±)'] = round(s['ci95'], 3)
    rows.append(row)
t7 = pd.DataFrame(rows)
mean_row = {'Endpoint': 'OVERALL MEAN'}
for model in ['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']:
    means = [summary[t][model]['mean'] for t in TASKS]
    cis = [summary[t][model]['ci95'] for t in TASKS]
    mean_row[f'{model} (mean AUC)'] = round(np.mean(means), 3)
    mean_row[f'{model} (95% CI ±)'] = round(np.mean(cis), 3)
t7 = pd.concat([t7, pd.DataFrame([mean_row])], ignore_index=True)
t7.to_csv('/home/claude/project/tables/table7_cv_summary.csv', index=False)
print(t7.to_string(index=False))

# Figure 9: CV performance with error bars, and per-fold box plots
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
x = np.arange(len(TASKS))
width = 0.2
colors = {'RandomForest': '#4C72B0', 'XGBoost': '#DD8452', 'Single_MLP': '#55A868', 'Deep_Ensemble': '#8172B2'}
for i, model in enumerate(['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']):
    means = [summary[t][model]['mean'] for t in TASKS]
    cis = [summary[t][model]['ci95'] for t in TASKS]
    ax.bar(x + (i-1.5)*width, means, width, yerr=cis, capsize=2, label=model.replace('_',' '),
           color=colors[model], edgecolor='black', linewidth=0.4, error_kw={'linewidth': 0.8})
ax.set_xticks(x); ax.set_xticklabels(TASKS, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('5-fold CV ROC-AUC (mean ± 95% CI)')
ax.set_ylim(0.4, 1.0)
ax.set_title('A. Cross-validated performance with 95% confidence intervals', fontweight='bold', fontsize=10.5)
ax.legend(fontsize=7.5, ncol=2)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
all_vals = {model: [] for model in ['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']}
for t in TASKS:
    for model in all_vals:
        all_vals[model].extend(summary[t][model]['values'])
bp = ax2.boxplot([all_vals[m] for m in all_vals], labels=['Random\nForest','XGBoost','Single\nMLP','Deep\nEnsemble'],
                  patch_artist=True, widths=0.55)
for patch, model in zip(bp['boxes'], all_vals.keys()):
    patch.set_facecolor(colors[model]); patch.set_alpha(0.7)
ax2.set_ylabel('Fold-level ROC-AUC (all endpoints pooled, 60 folds each)')
ax2.set_title('B. Distribution of per-fold AUC across all endpoints', fontweight='bold', fontsize=10.5)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig9_cv_validation.png', dpi=300, bbox_inches='tight')
print("\nSaved fig9_cv_validation.png")
