import numpy as np, pickle, pandas as pd

TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
         'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
FULL_NAMES = {
 'NR-AR':'Androgen Receptor','NR-AR-LBD':'Androgen Receptor (LBD)','NR-AhR':'Aryl Hydrocarbon Receptor',
 'NR-Aromatase':'Aromatase','NR-ER':'Estrogen Receptor','NR-ER-LBD':'Estrogen Receptor (LBD)',
 'NR-PPAR-gamma':'PPAR-gamma','SR-ARE':'Antioxidant Response Element','SR-ATAD5':'ATAD5 (genotoxicity)',
 'SR-HSE':'Heat Shock Response','SR-MMP':'Mitochondrial Membrane Potential','SR-p53':'p53 (DNA damage)'}

data = np.load('/home/claude/project/data/featurized.npz')
Y, mask = data['Y'], data['mask']

with open('/home/claude/project/models/ensemble_results.pkl','rb') as f: ens = pickle.load(f)
with open('/home/claude/project/models/baseline_results.pkl','rb') as f: base = pickle.load(f)
with open('/home/claude/project/models/uncertainty_analysis.pkl','rb') as f: ua = pickle.load(f)
with open('/home/claude/project/models/shap_results.pkl','rb') as f: shap_res = pickle.load(f)

# TABLE 1: Dataset characteristics
rows = []
for t in TASKS:
    n_total = mask[:,TASKS.index(t)].sum()
    n_pos = Y[mask[:,TASKS.index(t)], TASKS.index(t)].sum()
    rows.append({
        'Endpoint': t, 'Full name': FULL_NAMES[t], 'N labeled': int(n_total),
        'N positive': int(n_pos), 'Positive rate (%)': round(100*n_pos/n_total,2),
        'Missing (%)': round(100*(1-mask[:,TASKS.index(t)].mean()),2)
    })
t1 = pd.DataFrame(rows)
t1.to_csv('/home/claude/project/tables/table1_dataset.csv', index=False)
print("Table 1:\n", t1.to_string(index=False))

# TABLE 2: Model performance comparison
rows = []
for t in TASKS:
    rows.append({
        'Endpoint': t,
        'Random Forest': round(base['RandomForest'][t],3),
        'XGBoost': round(base['XGBoost'][t],3),
        'Single MLP': round(base['Single_MLP'][t],3),
        'Deep Ensemble (ours)': round(ens['results'][t]['auc'],3),
        'Ensemble epistemic std (mean)': round(float(ens['results'][t]['y_pred_std'].mean()),4)
    })
t2 = pd.DataFrame(rows)
mean_row = {'Endpoint':'MEAN', 'Random Forest': round(np.mean(list(base['RandomForest'].values())),3),
            'XGBoost': round(np.mean(list(base['XGBoost'].values())),3),
            'Single MLP': round(np.mean(list(base['Single_MLP'].values())),3),
            'Deep Ensemble (ours)': round(np.nanmean([ens['results'][t]['auc'] for t in TASKS]),3),
            'Ensemble epistemic std (mean)': round(np.mean([ens['results'][t]['y_pred_std'].mean() for t in TASKS]),4)}
t2 = pd.concat([t2, pd.DataFrame([mean_row])], ignore_index=True)
t2.to_csv('/home/claude/project/tables/table2_performance.csv', index=False)
print("\nTable 2:\n", t2.to_string(index=False))

# TABLE 3: Calibration metrics + selective prediction summary
rows = [{
    'Metric': 'Expected Calibration Error (ECE)', 'Value': round(ua['ece'],4)
},{
    'Metric': 'Brier score (pooled, 12 endpoints)', 'Value': round(ua['brier'],4)
}]
for i, frac in enumerate(ua['reject_fractions']):
    rows.append({'Metric': f'Retained AUC @ {int(frac*100)}% rejected (highest epistemic std)',
                 'Value': round(ua['selective_auc'][i],4)})
t3 = pd.DataFrame(rows)
t3.to_csv('/home/claude/project/tables/table3_calibration.csv', index=False)
print("\nTable 3:\n", t3.to_string(index=False))

# TABLE 4: Per-task uncertainty-stratified AUC
rows = []
for t in TASKS:
    s = ua['per_task_stats'][t]
    rows.append({'Endpoint': t, 'Overall AUC': round(s['overall_auc'],3),
                 'AUC (below-median epistemic std)': round(s['auc_low_unc'],3),
                 'AUC (above-median epistemic std)': round(s['auc_high_unc'],3),
                 'Delta (high-low)': round(s['auc_high_unc']-s['auc_low_unc'],3)})
t4 = pd.DataFrame(rows)
t4.to_csv('/home/claude/project/tables/table4_uncertainty_strata.csv', index=False)
print("\nTable 4:\n", t4.to_string(index=False))

# TABLE 5: Top SHAP features per focus endpoint
rows = []
for task in ['SR-MMP', 'NR-AhR']:
    for name, val, idx in shap_res[task]['top_features'][:8]:
        feat_type = 'Morgan-FP substructure' if 'FP_' in name else 'Physicochemical descriptor'
        rows.append({'Endpoint': task, 'Feature': name, 'Type': feat_type, 'Mean |SHAP|': round(float(val),4)})
t5 = pd.DataFrame(rows)
t5.to_csv('/home/claude/project/tables/table5_shap_features.csv', index=False)
print("\nTable 5:\n", t5.to_string(index=False))

print("\nAll 5 tables saved to /home/claude/project/tables/")
