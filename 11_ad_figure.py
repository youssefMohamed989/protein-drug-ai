import pandas as pd, matplotlib.pyplot as plt, numpy as np

plt.rcParams.update({'font.size': 10})
df = pd.read_csv('/home/claude/project/tables/table6_external_validation.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
colors = df['Correct'].map({'Yes': '#2A9D8F', 'No': '#E63946'})
markers_ext = df['In training set'] == 'No (external)'
ax.scatter(df.loc[markers_ext, 'Max Tanimoto to training set'], df.loc[markers_ext, 'Predicted P(toxic)'],
           c=colors[markers_ext], s=110, marker='D', edgecolor='black', linewidth=1, label='True external', zorder=3)
ax.scatter(df.loc[~markers_ext, 'Max Tanimoto to training set'], df.loc[~markers_ext, 'Predicted P(toxic)'],
           c=colors[~markers_ext], s=90, marker='o', edgecolor='black', linewidth=1, label='In training set', zorder=3, alpha=0.85)
ax.axvline(0.4, color='gray', linestyle='--', linewidth=1.3, label='AD threshold (Tanimoto = 0.4)')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
for _, row in df.iterrows():
    if row['Max Tanimoto to training set'] < 0.5 or row['Correct'] == 'No':
        ax.annotate(row['Compound'].split('(')[0].strip()[:18], (row['Max Tanimoto to training set'], row['Predicted P(toxic)']),
                    fontsize=6.5, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel('Max Tanimoto similarity to nearest Tox21 training compound')
ax.set_ylabel('Ensemble-predicted P(toxic)')
ax.set_title('A. Applicability domain: prediction vs. chemical similarity\nto training data', fontweight='bold', fontsize=10)
legend_elems = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#2A9D8F', markersize=9, label='Correct prediction'),
                plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#E63946', markersize=9, label='Incorrect prediction'),
                plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label='True external compound'),
                plt.Line2D([0],[0], color='gray', linestyle='--', label='AD threshold (0.4)')]
ax.legend(handles=legend_elems, fontsize=7, loc='center left')
ax.grid(alpha=0.25)
ax.set_xlim(0.2, 1.05); ax.set_ylim(-0.05, 1.05)

ax2 = axes[1]
inside = df[df['AD status'].str.startswith('Inside')]
outside = df[df['AD status'].str.startswith('Outside')]
acc_inside = (inside['Correct'] == 'Yes').mean() * 100
acc_outside = (outside['Correct'] == 'Yes').mean() * 100 if len(outside) else 0
bars = ax2.bar(['Inside AD\n(Tanimoto \u2265 0.4)\nn={}'.format(len(inside)), 'Outside AD\n(Tanimoto < 0.4)\nn={}'.format(len(outside))],
               [acc_inside, acc_outside], color=['#2A9D8F', '#E63946'], edgecolor='black', linewidth=1, width=0.55)
for bar, v in zip(bars, [acc_inside, acc_outside]):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax2.set_ylabel('Prediction accuracy on external reference set (%)')
ax2.set_title('B. Accuracy stratified by applicability domain', fontweight='bold', fontsize=10)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig10_applicability_domain.png', dpi=300, bbox_inches='tight')
print("Saved fig10_applicability_domain.png")
