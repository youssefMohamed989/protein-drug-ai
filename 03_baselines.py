"""
Baseline comparison: Random Forest, XGBoost, single (non-ensemble) MLP.
Establishes that ensemble approach doesn't sacrifice accuracy for uncertainty benefits.
"""
import numpy as np, pickle, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

np.random.seed(42)
TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

data = np.load('/home/claude/project/data/featurized.npz')
X, Y, mask = data['X'], data['Y'], data['mask']

with open('/home/claude/project/models/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)
idx_train, idx_val, idx_test = ens['idx_train'], ens['idx_val'], ens['idx_test']

X_train, X_test = X[idx_train], X[idx_test]
Y_train, Y_test = Y[idx_train], Y[idx_test]
M_train, M_test = mask[idx_train], mask[idx_test]

baseline_results = {'RandomForest': {}, 'XGBoost': {}, 'Single_MLP': {}}

for t_idx, task in enumerate(TASKS):
    tr_m, te_m = M_train[:, t_idx], M_test[:, t_idx]
    Xt, yt = X_train[tr_m], Y_train[tr_m, t_idx]
    Xte, yte = X_test[te_m], Y_test[te_m, t_idx]

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight='balanced')
    rf.fit(Xt, yt)
    p_rf = rf.predict_proba(Xte)[:, 1]
    baseline_results['RandomForest'][task] = roc_auc_score(yte, p_rf)

    pos_w = (len(yt) - yt.sum()) / max(yt.sum(), 1)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         scale_pos_weight=pos_w, eval_metric='logloss', n_jobs=-1, random_state=42)
    xgb.fit(Xt, yt)
    p_xgb = xgb.predict_proba(Xte)[:, 1]
    baseline_results['XGBoost'][task] = roc_auc_score(yte, p_xgb)

    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=200, random_state=0, early_stopping=True)
    mlp.fit(Xt, yt)
    p_mlp = mlp.predict_proba(Xte)[:, 1]
    baseline_results['Single_MLP'][task] = roc_auc_score(yte, p_mlp)

    print(f"[{task}] RF={baseline_results['RandomForest'][task]:.3f}  "
          f"XGB={baseline_results['XGBoost'][task]:.3f}  "
          f"MLP1={baseline_results['Single_MLP'][task]:.3f}  "
          f"Ensemble={ens['results'][task]['auc']:.3f}")

with open('/home/claude/project/models/baseline_results.pkl', 'wb') as f:
    pickle.dump(baseline_results, f)

for m in baseline_results:
    print(m, "mean AUC:", np.mean(list(baseline_results[m].values())))
print("Ensemble mean AUC:", np.nanmean([ens['results'][t]['auc'] for t in TASKS]))
