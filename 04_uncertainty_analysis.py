"""
Calibration analysis + selective prediction (reject-based-on-uncertainty) curves.
Core novelty claim: high epistemic-uncertainty predictions are less reliable;
rejecting them improves accuracy on the retained set -- clinically actionable triage.
"""
import numpy as np, pickle
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

with open('/home/claude/project/models/ensemble_results.pkl', 'rb') as f:
    ens = pickle.load(f)
results = ens['results']

# --- Reliability diagram data (calibration curve), pooled across all tasks ---
all_true, all_pred, all_std = [], [], []
for task in TASKS:
    r = results[task]
    all_true.append(r['y_true']); all_pred.append(r['y_pred_mean']); all_std.append(r['y_pred_std'])
all_true = np.concatenate(all_true); all_pred = np.concatenate(all_pred); all_std = np.concatenate(all_std)

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_ids = np.digitize(all_pred, bin_edges[1:-1])
calib_conf, calib_acc, calib_count = [], [], []
for b in range(n_bins):
    m = bin_ids == b
    if m.sum() > 0:
        calib_conf.append(all_pred[m].mean())
        calib_acc.append(all_true[m].mean())
        calib_count.append(m.sum())
    else:
        calib_conf.append(np.nan); calib_acc.append(np.nan); calib_count.append(0)

ece = np.nansum([c/len(all_pred) * abs(a - conf) for c, a, conf in zip(calib_count, calib_acc, calib_conf)])
brier = brier_score_loss(all_true, all_pred)
print(f"Expected Calibration Error (ECE): {ece:.4f}")
print(f"Brier score (pooled): {brier:.4f}")

# --- Selective prediction: rank by epistemic uncertainty (std), reject top-k%, measure AUC on rest ---
reject_fractions = np.linspace(0, 0.5, 11)
selective_auc = []
for frac in reject_fractions:
    threshold = np.quantile(all_std, 1 - frac) if frac > 0 else np.inf
    keep = all_std <= threshold
    if len(np.unique(all_true[keep])) > 1:
        auc = roc_auc_score(all_true[keep], all_pred[keep])
    else:
        auc = np.nan
    selective_auc.append(auc)
    print(f"Reject top {frac*100:.0f}% uncertain -> retained AUC = {auc:.4f} (n={keep.sum()})")

# --- Per-task calibration + uncertainty-stratified accuracy ---
per_task_stats = {}
for task in TASKS:
    r = results[task]
    median_std = np.median(r['y_pred_std'])
    low_unc = r['y_pred_std'] <= median_std
    high_unc = ~low_unc
    auc_low = roc_auc_score(r['y_true'][low_unc], r['y_pred_mean'][low_unc]) if len(np.unique(r['y_true'][low_unc])) > 1 else np.nan
    auc_high = roc_auc_score(r['y_true'][high_unc], r['y_pred_mean'][high_unc]) if len(np.unique(r['y_true'][high_unc])) > 1 else np.nan
    per_task_stats[task] = {'auc_low_unc': auc_low, 'auc_high_unc': auc_high, 'overall_auc': r['auc']}

with open('/home/claude/project/models/uncertainty_analysis.pkl', 'wb') as f:
    pickle.dump({
        'calib_conf': calib_conf, 'calib_acc': calib_acc, 'calib_count': calib_count,
        'ece': ece, 'brier': brier,
        'reject_fractions': reject_fractions, 'selective_auc': selective_auc,
        'per_task_stats': per_task_stats,
        'all_true': all_true, 'all_pred': all_pred, 'all_std': all_std
    }, f)
print("\nPer-task low-vs-high uncertainty AUC split:")
for t, s in per_task_stats.items():
    print(f"  {t}: overall={s['overall_auc']:.3f} low-unc={s['auc_low_unc']:.3f} high-unc={s['auc_high_unc']:.3f}")
