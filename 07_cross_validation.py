"""
Repeated 5-fold stratified cross-validation for the deep ensemble and all baselines,
reporting mean +/- 95% CI ROC-AUC per endpoint (addresses single-split limitation).
"""
import numpy as np, pickle, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy import stats
warnings.filterwarnings('ignore')

np.random.seed(42)
TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

data = np.load('/home/claude/project/data/featurized.npz')
X, Y, mask = data['X'], data['Y'], data['mask']

N_FOLDS = 5
N_ENSEMBLE_CV = 4  # reduced ensemble size per fold for tractable runtime; 5 folds x 4 models = 20 fits/task

cv_results = {t: {'RandomForest': [], 'XGBoost': [], 'Single_MLP': [], 'Deep_Ensemble': []} for t in TASKS}

for t_idx, task in enumerate(TASKS):
    m = mask[:, t_idx]
    Xt, yt = X[m], Y[m, t_idx]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_num = 0
    for train_idx, test_idx in skf.split(Xt, yt):
        fold_num += 1
        Xtr, Xte = Xt[train_idx], Xt[test_idx]
        ytr, yte = yt[train_idx], yt[test_idx]
        if len(np.unique(yte)) < 2:
            continue

        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight='balanced')
        rf.fit(Xtr, ytr)
        cv_results[task]['RandomForest'].append(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))

        pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             scale_pos_weight=pos_w, eval_metric='logloss', n_jobs=-1, random_state=42)
        xgb.fit(Xtr, ytr)
        cv_results[task]['XGBoost'].append(roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1]))

        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=200, random_state=0, early_stopping=True)
        mlp.fit(Xtr, ytr)
        cv_results[task]['Single_MLP'].append(roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1]))

        ens_preds = []
        for e in range(N_ENSEMBLE_CV):
            clf = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-4, max_iter=200,
                                 random_state=e * 17 + 1, early_stopping=True, n_iter_no_change=10)
            clf.fit(Xtr, ytr)
            ens_preds.append(clf.predict_proba(Xte)[:, 1])
        ens_mean = np.mean(ens_preds, axis=0)
        cv_results[task]['Deep_Ensemble'].append(roc_auc_score(yte, ens_mean))

        print(f"[{task}] fold {fold_num}/{N_FOLDS} done")

# Summarize: mean, std, 95% CI (t-distribution) per task/model
summary = {}
for task in TASKS:
    summary[task] = {}
    for model in ['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']:
        vals = np.array(cv_results[task][model])
        n = len(vals)
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ci = stats.t.ppf(0.975, n - 1) * se if n > 1 else 0.0
        summary[task][model] = {'mean': mean, 'std': vals.std(ddof=1) if n > 1 else 0.0,
                                 'ci95': ci, 'n_folds': n, 'values': vals.tolist()}

with open('/home/claude/project/models/cv_results.pkl', 'wb') as f:
    pickle.dump({'cv_results': cv_results, 'summary': summary}, f)

print("\n=== 5-fold CV summary (mean AUC ± 95% CI) ===")
for task in TASKS:
    line = f"{task}: "
    for model in ['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']:
        s = summary[task][model]
        line += f"{model}={s['mean']:.3f}±{s['ci95']:.3f}  "
    print(line)

print("\n=== Overall mean across endpoints ===")
for model in ['RandomForest', 'XGBoost', 'Single_MLP', 'Deep_Ensemble']:
    means = [summary[t][model]['mean'] for t in TASKS]
    print(f"{model}: {np.mean(means):.4f} (SD across endpoints {np.std(means):.4f})")
