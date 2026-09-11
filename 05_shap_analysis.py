"""
SHAP-based interpretability using XGBoost surrogate (SHAP TreeExplainer is fast/exact),
mapped back to interpretable descriptor names + top Morgan-fingerprint bit substructures.
"""
import numpy as np, pickle, warnings
from xgboost import XGBClassifier
import shap
warnings.filterwarnings('ignore')

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
DESC_NAMES = ['MolWt','LogP','NumHDonors','NumHAcceptors','TPSA',
              'NumRotatableBonds','NumAromaticRings','NumRings','FractionCSP3','HeavyAtomCount']

data = np.load('/home/claude/project/data/featurized.npz')
X, Y, mask = data['X'], data['Y'], data['mask']
feature_names = [f'FP_{i}' for i in range(1024)] + DESC_NAMES

with open('/home/claude/project/models/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)
idx_train, idx_test = ens['idx_train'], ens['idx_test']

# Focus SHAP on two representative endpoints: best (SR-MMP, mitochondrial toxicity)
# and clinically critical (NR-AhR, xenobiotic receptor activation)
focus_tasks = ['SR-MMP', 'NR-AhR']
shap_results = {}

for task in focus_tasks:
    t_idx = TASKS.index(task)
    tr_m = mask[idx_train][:, t_idx]
    Xt, yt = X[idx_train][tr_m], Y[idx_train][tr_m, t_idx]

    pos_w = (len(yt) - yt.sum()) / max(yt.sum(), 1)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         scale_pos_weight=pos_w, eval_metric='logloss', n_jobs=-1, random_state=42)
    xgb.fit(Xt, yt)

    explainer = shap.TreeExplainer(xgb)
    sample_idx = np.random.RandomState(0).choice(len(Xt), size=min(300, len(Xt)), replace=False)
    X_sample = Xt[sample_idx]
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = [(feature_names[i], mean_abs_shap[i], i) for i in top_idx]

    shap_results[task] = {
        'shap_values': shap_values, 'X_sample': X_sample,
        'top_features': top_features, 'feature_names': feature_names
    }
    print(f"\n[{task}] Top 10 SHAP features:")
    for name, val, i in top_features[:10]:
        print(f"    {name}: mean|SHAP| = {val:.4f}")

with open('/home/claude/project/models/shap_results.pkl', 'wb') as f:
    pickle.dump(shap_results, f)
print("\nSaved shap_results.pkl")
