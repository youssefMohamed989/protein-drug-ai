import matplotlib.pyplot as plt
import numpy as np
import pickle
import umap

plt.rcParams.update({'font.size': 10})
np.random.seed(42)

data = np.load('/home/claude/project/data/featurized.npz')
X = data['X']

with open('/home/claude/project/models/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)
idx_test = ens['idx_test']
X_test = X[idx_test]

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
results = ens['results']

# Use SR-MMP task's test-subset predictions (has mask applied so indices differ);
# recompute mapping to X_test using mask indices consistent with training script
mask = data['mask']
Y = data['Y']
t_idx = TASKS.index('SR-MMP')
test_mask_t = mask[idx_test][:, t_idx]
X_sub = X_test[test_mask_t]
r = results['SR-MMP']

print("Fitting UMAP on", X_sub.shape[0], "test compounds...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard', random_state=42)
embedding = reducer.fit_transform(X_sub[:, :1024])  # fingerprint bits only, Jaccard-appropriate

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# Panel A: colored by true label
ax = axes[0]
sc0 = ax.scatter(embedding[:, 0], embedding[:, 1], c=r['y_true'], cmap='coolwarm',
                  s=12, alpha=0.75, edgecolor='none')
ax.set_title('A. Ground-truth SR-MMP toxicity label', fontweight='bold', fontsize=10)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
cb0 = plt.colorbar(sc0, ax=ax, shrink=0.8, ticks=[0,1])
cb0.set_ticklabels(['Non-toxic', 'Toxic'])

# Panel B: colored by predicted probability
ax2 = axes[1]
sc1 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=r['y_pred_mean'], cmap='magma',
                   s=12, alpha=0.75, edgecolor='none')
ax2.set_title('B. Ensemble-predicted toxicity probability', fontweight='bold', fontsize=10)
ax2.set_xlabel('UMAP-1'); ax2.set_ylabel('UMAP-2')
plt.colorbar(sc1, ax=ax2, shrink=0.8, label='P(toxic)')

# Panel C: colored by epistemic uncertainty
ax3 = axes[2]
sc2 = ax3.scatter(embedding[:, 0], embedding[:, 1], c=r['y_pred_std'], cmap='viridis',
                   s=12, alpha=0.75, edgecolor='none')
ax3.set_title('C. Epistemic uncertainty (ensemble std)', fontweight='bold', fontsize=10)
ax3.set_xlabel('UMAP-1'); ax3.set_ylabel('UMAP-2')
plt.colorbar(sc2, ax=ax3, shrink=0.8, label='Prediction std')

fig.suptitle('Chemical-space projection of the SR-MMP test set (UMAP on Morgan fingerprints, Jaccard metric)',
             fontweight='bold', fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig('/home/claude/project/figures/fig7_chemspace.png', dpi=300, bbox_inches='tight')
print("Saved fig7_chemspace.png")
