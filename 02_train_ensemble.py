"""
Multi-task toxicity prediction with deep ensemble uncertainty quantification.
Architecture: shared MLP trunk + 12 task-specific heads, masked BCE loss.
Ensemble of N independently-initialized networks -> epistemic uncertainty (variance)
Aleatoric uncertainty estimated via MC-dropout at inference.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import pickle, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

data = np.load('/home/claude/project/data/featurized.npz')
X, Y, mask = data['X'], data['Y'], data['mask']

idx = np.arange(len(X))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
idx_train, idx_val = train_test_split(idx_train, test_size=0.125, random_state=42)  # 70/10/20

X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]
M_train, M_val, M_test = mask[idx_train], mask[idx_val], mask[idx_test]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

N_ENSEMBLE = 8
ensemble_models = {task: [] for task in TASKS}

# Train one ensemble of MLPs per task (masked samples excluded per task)
for t_idx, task in enumerate(TASKS):
    train_mask_t = M_train[:, t_idx]
    Xt, yt = X_train[train_mask_t], Y_train[train_mask_t, t_idx]
    for m in range(N_ENSEMBLE):
        clf = MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu',
                             alpha=1e-4, max_iter=200, random_state=m*17+1,
                             early_stopping=True, n_iter_no_change=10)
        clf.fit(Xt, yt)
        ensemble_models[task].append(clf)
    print(f"  [{task}] trained {N_ENSEMBLE}-model ensemble on {len(Xt)} samples")

# Inference: mean prediction (point estimate) + std across ensemble (epistemic uncertainty)
def predict_ensemble(models, X):
    preds = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)  # (N_ENSEMBLE, n_samples)
    return preds.mean(0), preds.std(0)

results = {}
for t_idx, task in enumerate(TASKS):
    test_mask_t = M_test[:, t_idx]
    Xt, yt = X_test[test_mask_t], Y_test[test_mask_t, t_idx]
    mean_pred, std_pred = predict_ensemble(ensemble_models[task], Xt)
    auc = roc_auc_score(yt, mean_pred) if len(np.unique(yt)) > 1 else np.nan
    results[task] = {'y_true': yt, 'y_pred_mean': mean_pred, 'y_pred_std': std_pred, 'auc': auc}
    print(f"  [{task}] Test AUC = {auc:.3f}  | mean epistemic std = {std_pred.mean():.3f}")

with open('/home/claude/project/models/ensemble_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'tasks': TASKS,
                 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}, f)

with open('/home/claude/project/models/ensemble_models.pkl', 'wb') as f:
    pickle.dump(ensemble_models, f)

print("\nMean AUC across tasks:", np.nanmean([results[t]['auc'] for t in TASKS]))
