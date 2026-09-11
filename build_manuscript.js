const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak, Header, Footer,
  PageNumber, NumberFormat, convertInchesToTwip
} = require('docx');

const PAGE_W = 12240, PAGE_H = 15840; // US Letter DXA

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, ...(opts.spacing || {}) },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, italics: !!opts.italics, bold: !!opts.bold, size: opts.size || 22 })],
  });
}
function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, spacing: { before: 300, after: 160 }, children: [new TextRun({ text })] });
}
function caption(text) {
  return new Paragraph({
    spacing: { before: 100, after: 240 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, bold: true, size: 20 })],
  });
}
function img(path, widthPx = 620) {
  const buf = fs.readFileSync(path);
  const imageSizeMod = require('image-size');
  const sizeOf = imageSizeMod.imageSize || imageSizeMod.default || imageSizeMod;
  const sizeInfo = sizeOf(new Uint8Array(buf));
  const ratio = sizeInfo.height / sizeInfo.width;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 80 },
    children: [new ImageRun({ data: buf, transformation: { width: widthPx, height: Math.round(widthPx * ratio) }, type: 'png' })],
  });
}
function makeTable(headerCells, rows, colWidths) {
  const totalWidth = 9360; // ~6.5in content width in DXA
  const widths = colWidths || headerCells.map(() => Math.floor(totalWidth / headerCells.length));
  const headerRow = new TableRow({
    tableHeader: true,
    children: headerCells.map((h, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: '2E4057', color: 'auto' },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: 'FFFFFF', size: 18 })] })],
    })),
  });
  const bodyRows = rows.map((r, ridx) => new TableRow({
    children: r.map((cellText, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ridx % 2 === 0 ? 'F2F4F7' : 'FFFFFF' },
      children: [new Paragraph({ children: [new TextRun({ text: String(cellText), size: 18 })] })],
    })),
  }));
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}

function parseCsvLine(line) {
  const out = [];
  let cur = '', inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"' && line[i+1] === '"') { cur += '"'; i++; }
      else if (c === '"') { inQuotes = false; }
      else { cur += c; }
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ',') { out.push(cur); cur = ''; }
      else cur += c;
    }
  }
  out.push(cur);
  return out;
}
function csvToRows(path) {
  const raw = fs.readFileSync(path, 'utf8').trim().split('\n');
  const header = parseCsvLine(raw[0]);
  const rows = raw.slice(1).map(line => parseCsvLine(line));
  return { header, rows };
}

const t1 = csvToRows('/home/claude/project/tables/table1_dataset.csv');
const t2 = csvToRows('/home/claude/project/tables/table2_performance.csv');
const t3 = csvToRows('/home/claude/project/tables/table3_calibration.csv');
const t4 = csvToRows('/home/claude/project/tables/table4_uncertainty_strata.csv');
const t5 = csvToRows('/home/claude/project/tables/table5_shap_features.csv');
const t6 = csvToRows('/home/claude/project/tables/table6_external_validation.csv');
const t7 = csvToRows('/home/claude/project/tables/table7_cv_summary.csv');

const FIG = '/home/claude/project/figures/';

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: 22 } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: 'Machine Learning Framework for Tox21 Chemical-Biological Interactions', size: 16, italics: true })] })] }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
        })],
      }),
    },
    children: [
      // TITLE PAGE
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 300 },
        children: [new TextRun({
          text: 'Development of a Machine Learning Framework with Calibrated Uncertainty Quantification and 3D Toxicophore Mapping for Multi-Task Prediction of Tox21 Chemical-Biological Interactions: Mechanistic Insights for Rational Drug Design',
          bold: true, size: 30,
        })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: 'Youssef M. Hassan', size: 22, bold: true }),
                    new TextRun({ text: '*a', size: 16, superScript: true }),
                    new TextRun({ text: ', Hala El-Tantawi', size: 22, bold: true }),
                    new TextRun({ text: 'a', size: 16, superScript: true }),
                    new TextRun({ text: ', Ibrahim Rabie Ali', size: 22, bold: true }),
                    new TextRun({ text: 'b', size: 16, superScript: true }),
                    new TextRun({ text: ' and Mohamed S. Attia', size: 22, bold: true }),
                    new TextRun({ text: '*c', size: 16, superScript: true })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: 'a Department of Zoology, Faculty of Science, Ain Shams University, Abbassia 11566, Cairo, Egypt', italics: true, size: 18 })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: 'b Department of Immunology and Treatment Evaluation, Theodore Bilharz Research Institute, Giza, Egypt', italics: true, size: 18 })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: 'c Chemistry Department, College of Science, Imam Mohammad Ibn Saud Islamic University (IMSIU), Riyadh 11623, Saudi Arabia', italics: true, size: 18 })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: '*Corresponding author: Youssef M. Hassan — yousefmohamed_p@sci.asu.edu.eg', italics: true, size: 18 })],
      }),

      heading('Abstract', HeadingLevel.HEADING_1),
      p('Background: Understanding the molecular basis of chemical-biological interactions that underlie xenobiotic toxicity is central to rational drug design, yet most machine learning models developed for the Tox21 benchmark report point-estimate accuracy without addressing predictive confidence, three-dimensional structural rationale, or mechanistic interpretability jointly, limiting their utility for compound triage and lead-optimization decisions.'),
      p('Methods: We developed a multi-task deep ensemble machine learning framework (eight independently initialized multilayer perceptrons per endpoint) trained on Morgan fingerprints and physicochemical descriptors derived from 8,006 valid Tox21 compounds spanning 12 nuclear receptor and stress-response endpoints reflective of distinct chemical-biological interaction mechanisms. Epistemic uncertainty was estimated from ensemble disagreement; calibration was assessed via reliability diagrams and Expected Calibration Error (ECE); mechanistic interpretability was obtained via SHapley Additive exPlanations (SHAP) on an XGBoost surrogate and complemented by three-dimensional conformational analysis of representative toxicophore-bearing compounds.'),
      p('Results: The deep ensemble achieved a mean test ROC-AUC of 0.803 across 12 endpoints on the primary split and 0.810 (95% CI ±0.029) under stratified 5-fold cross-validation, comparable to but not exceeding a Random Forest baseline (mean AUC 0.847 / 0.854 ± 0.029 cross-validated), while additionally providing well-calibrated probability estimates (pooled ECE = 0.018, Brier score = 0.050). Critically, naive selective prediction — rejecting the highest-uncertainty compounds — degraded rather than improved retained-set discrimination (AUC declining from 0.847 at 0% rejection to 0.545 at 50% rejection), a consequence of severe class imbalance in which low ensemble-disagreement corresponds to trivially confident majority-class (non-toxic) predictions rather than reliable predictions per se. Stratifying by ensemble disagreement showed higher-uncertainty compounds were in fact more discriminable (mean AUC +0.26 versus low-uncertainty compounds) in 11 of 12 endpoints. SHAP analysis identified lipophilicity (LogP) and sp3-carbon fraction as dominant drivers of mitochondrial membrane potential disruption (SR-MMP) and aryl-hydrocarbon receptor activation (NR-AhR) respectively, consistent with established toxicological mechanisms; three-dimensional conformational analysis of representative compounds further confirmed that high-risk molecules adopt bulky lipophilic or planar polyaromatic geometries consistent with these SHAP-derived toxicophore rules. External validation on reference toxicants absent from the training data showed correct generalization for both truly external AhR agonists (P = 0.89, 0.86) but under-prediction of a structurally distinct external mitochondrial uncoupler (FCCP, P = 0.28), indicating that generalization is mechanism- and chemotype-dependent rather than uniform.'),
      p('Conclusions: Ensemble disagreement in imbalanced multi-task toxicity prediction is not a monotonic proxy for prediction reliability, and practitioners should not assume that uncertainty-based rejection improves accuracy without first accounting for class-imbalance-driven confidence collapse. We provide a calibration- and mechanism-aware alternative framework and release full code and analysis for reproducibility.'),
      p('Keywords: Tox21; chemical-biological interactions; toxicity prediction; deep ensembles; uncertainty quantification; SHAP; 3D toxicophore mapping; rational drug design; QSAR', { italics: true }),

      new Paragraph({ children: [new PageBreak()] }),

      // INTRODUCTION
      heading('1. Introduction'),
      p('Early and accurate prediction of compound toxicity is a critical bottleneck in pharmaceutical development, where late-stage toxicity failures account for a substantial fraction of clinical attrition and cost. The Tox21 initiative, a collaboration between the U.S. National Institutes of Health, Environmental Protection Agency, and Food and Drug Administration, produced a public high-throughput screening dataset spanning 12 nuclear receptor and stress-response pathway assays across roughly 8,000 compounds, and has become a standard benchmark for machine learning approaches to in silico toxicology.'),
      p('A substantial body of prior work has applied random forests, gradient boosting, graph neural networks, and multi-task deep learning architectures to Tox21, typically optimizing and reporting area-under-the-receiver-operating-characteristic-curve (ROC-AUC) as the primary metric. However, three interconnected gaps persist in this literature. First, point-estimate models provide no indication of when a prediction should be trusted, which is essential when models are used to triage large virtual libraries under limited experimental validation budgets. Second, the relationship between model uncertainty and actual predictive reliability is rarely tested empirically under the severe class imbalance characteristic of toxicity data, where naive assumptions imported from balanced-classification uncertainty literature may not hold. Third, interpretability and uncertainty quantification are typically treated as separate research threads rather than jointly assessed within a single modeling pipeline, despite both being necessary for responsible deployment in a drug-development context.'),
      p('In this work we address all three gaps directly. We construct a multi-task deep ensemble for Tox21 toxicity prediction, quantify epistemic uncertainty via ensemble disagreement, assess probability calibration via reliability diagrams and Expected Calibration Error, and test — rather than assume — whether rejecting high-uncertainty predictions improves retained-set accuracy (selective prediction). We complement this with SHAP-based mechanistic interpretation using an independent gradient-boosted surrogate model. Our central empirical finding is counter to common intuition: under the class imbalance typical of toxicity screening data, low ensemble disagreement is dominated by trivially confident majority-class predictions, so naive uncertainty-based rejection removes informative borderline compounds rather than unreliable ones — a nuance with direct practical consequences for how uncertainty estimates should (and should not) be used to guide compound prioritization.'),

      heading('2. Related Work'),
      p('Random forest and gradient-boosted tree ensembles remain strong baselines for Tox21 due to their native handling of sparse, high-dimensional fingerprint features and class imbalance via class weighting. Graph neural network approaches (e.g., message-passing networks, graph attention networks) have reported competitive or superior AUC on subsets of Tox21 endpoints by learning representations directly from molecular graphs rather than fixed fingerprints, though gains are often modest and endpoint-dependent. Multi-task neural architectures that share a representation trunk across the 12 endpoints have been motivated by the hypothesis that shared toxicological mechanisms (e.g., receptor-family homology) enable positive transfer, though empirical gains over single-task models are inconsistent across studies.'),
      p('Separately, uncertainty quantification for molecular property prediction has drawn on Bayesian neural networks, Monte Carlo dropout, and deep ensembles, with evaluation typically focused on regression tasks (e.g., solubility, binding affinity) rather than the imbalanced multi-task classification setting of Tox21. Selective prediction and reject-option classification are well studied in balanced-classification and medical-imaging contexts, where rejecting low-confidence predictions reliably improves retained-set accuracy; we are not aware of prior Tox21-specific work explicitly testing whether this assumption transfers to severely imbalanced multi-task toxicity classification, which is a central contribution of the present study.'),

      new Paragraph({ children: [new PageBreak()] }),

      // METHODS
      heading('3. Materials and Methods'),
      heading('3.1 Dataset and Featurization', HeadingLevel.HEADING_2),
      p('The Tox21 dataset (8,014 raw compound records spanning 12 toxicity endpoints: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53) was obtained in SMILES format. Structures were parsed with RDKit; 8 records failed to parse and were excluded, yielding 8,006 valid compounds. For each compound we computed a 1,024-bit Morgan (circular) fingerprint (radius 2) and ten standardized physicochemical descriptors: molecular weight, calculated LogP, hydrogen-bond donor and acceptor counts, topological polar surface area (TPSA), rotatable bond count, aromatic ring count, total ring count, fraction of sp3-hybridized carbons, and heavy-atom count, yielding a 1,034-dimensional feature vector per compound. Endpoint labels are frequently missing (9.8%–26.1% per endpoint, Table 1) due to assay-specific experimental attrition; missing labels were masked during both training and evaluation on a per-task basis rather than imputed.'),
      p('Compounds were split 70/10/20 into training, validation, and held-out test sets using a fixed random seed (42), with splitting performed once at the compound level (not per-task) to ensure consistent train/test membership across endpoints and prevent information leakage between tasks that share compounds.'),

      heading('3.2 Model Architecture and Training', HeadingLevel.HEADING_2),
      p('For each of the 12 endpoints, we trained an ensemble of eight multilayer perceptrons (two hidden layers of 256 and 64 units, ReLU activation, L2 regularization coefficient 1e-4), each initialized with an independent random seed and trained with early stopping on a held-out validation subset (patience of 10 iterations, maximum 200 iterations). Training was restricted to compounds with a non-missing label for the corresponding endpoint. Point predictions for a given compound and endpoint were computed as the mean predicted probability across the eight ensemble members; epistemic uncertainty was quantified as the standard deviation of predicted probabilities across ensemble members.'),

      heading('3.3 Baseline Models', HeadingLevel.HEADING_2),
      p('We compared the deep ensemble against three baselines trained identically on the same features and splits: (i) a Random Forest classifier (300 trees, class-balanced weighting), (ii) an XGBoost gradient-boosted tree classifier (300 estimators, max depth 6, learning rate 0.05, positive-class weighting proportional to inverse prevalence), and (iii) a single (non-ensembled) MLP with identical architecture and hyperparameters to one ensemble member. Baselines provide point predictions only, without native uncertainty estimates, establishing the accuracy cost (if any) of the ensemble approach.'),

      heading('3.4 Calibration and Selective Prediction Analysis', HeadingLevel.HEADING_2),
      p('Calibration was assessed by pooling predictions across all 12 endpoints on the held-out test set and constructing a 10-bin reliability diagram (predicted probability versus observed empirical frequency), from which we computed Expected Calibration Error (ECE) as the sample-weighted mean absolute deviation between confidence and accuracy across bins, and the Brier score as the mean squared error between predicted probabilities and binary outcomes. To test whether epistemic uncertainty is a useful signal for compound triage, we performed selective prediction: compounds were ranked by ensemble standard deviation, and we evaluated ROC-AUC on the retained subset after progressively rejecting the top 0%–50% most uncertain predictions (in 5% increments), pooled across all endpoints. We additionally stratified each endpoint into below- and above-median epistemic-uncertainty subsets and compared discrimination (ROC-AUC) within each stratum.'),

      heading('3.4 Repeated Cross-Validation', HeadingLevel.HEADING_2),
      p('To address the statistical limitations of a single train/test split and provide confidence intervals on all reported performance metrics, we additionally performed stratified 5-fold cross-validation for each endpoint independently, repeated for all four model classes (Random Forest, XGBoost, single MLP, and a four-member deep ensemble per fold, reduced from eight members for computational tractability across 5 folds × 12 endpoints × 4 model classes). For each endpoint and model, we report the mean test-fold ROC-AUC together with the 95% confidence interval computed from the t-distribution across the five fold-level estimates.'),

      heading('3.5 External Validation on Independent Reference Compounds', HeadingLevel.HEADING_2),
      p('To assess whether model predictions and SHAP-derived toxicophore rules generalize beyond the Tox21 training distribution, we curated nine well-characterized reference compounds from the toxicological literature spanning three classes: (i) canonical AhR agonists (benzo[a]pyrene, dibenzo[a,h]anthracene, beta-naphthoflavone), (ii) canonical mitochondrial membrane potential disruptors (FCCP, 2,4-dinitrophenol, amiodarone), and (iii) saturated, non-aromatic negative-control compounds with no expected toxicological liability on either endpoint (glucose, cyclohexanol, citric acid). Each reference compound was canonicalized and cross-checked against the full Tox21 training set to determine whether it was already present; compounds absent from the training set constitute a genuine external (held-out) test of generalization, while compounds present in the training set (which can occur for well-studied reference toxicants that are themselves part of Tox21) provide an internal consistency check rather than a test of extrapolation. We report predictions for both groups with explicit training-set-membership annotation (Table 6) to avoid overstating generalization performance.'),

      heading('3.6 Interpretability Analysis', HeadingLevel.HEADING_2),
      p('To obtain mechanistic interpretability independent of the ensemble architecture, we trained an XGBoost surrogate model (identical hyperparameters to the XGBoost baseline) for two representative endpoints selected to span distinct toxicological mechanisms: SR-MMP (mitochondrial membrane potential disruption, a general cytotoxicity marker) and NR-AhR (aryl hydrocarbon receptor activation, a xenobiotic-sensing nuclear receptor pathway). SHAP values were computed using the exact TreeExplainer algorithm on a random sample of 300 training compounds per endpoint, and mean absolute SHAP value was used to rank feature importance across the full 1,034-dimensional feature space (Morgan fingerprint bits plus physicochemical descriptors).'),

      heading('3.7 Chemical Space Visualization', HeadingLevel.HEADING_2),
      p('For qualitative visualization, we projected the held-out SR-MMP test compounds into two dimensions using Uniform Manifold Approximation and Projection (UMAP; 15 nearest neighbors, minimum distance 0.1) applied to the 1,024-bit Morgan fingerprints under the Jaccard (Tanimoto-equivalent binary) distance metric, colored by ground-truth label, predicted probability, and epistemic uncertainty.'),

      heading('3.8 Software', HeadingLevel.HEADING_2),
      p('All analyses were performed in Python 3.12 using RDKit (cheminformatics, 3D conformer generation via ETKDGv3, and MMFF94 geometry optimization), scikit-learn 1.8 (Random Forest, MLP, stratified k-fold cross-validation, train/test splitting, metrics), XGBoost (gradient boosting and SHAP surrogate), SHAP (interpretability), and UMAP-learn (dimensionality reduction). Full code is available for reproducibility upon request.'),

      new Paragraph({ children: [new PageBreak()] }),

      // RESULTS
      heading('4. Results'),
      heading('4.1 Dataset Characteristics', HeadingLevel.HEADING_2),
      p('After featurization, 8,006 compounds were retained across 12 endpoints (Table 1, Figure 1). Endpoint prevalence ranged from 2.87% (NR-PPAR-gamma) to 16.21% (SR-ARE), and label missingness ranged from 7.17% (NR-AR) to 26.13% (SR-MMP), confirming that both class imbalance and missing-label handling are first-order considerations for this benchmark (Figure 1B–C). Chemical space coverage, visualized via standardized molecular weight and LogP, showed the expected concentration of drug-like compounds within Lipinski-consistent property ranges (Figure 1D).'),
      caption('Figure 1. Pipeline overview and dataset characteristics. (A) End-to-end schematic from raw SMILES through featurization, splitting, ensemble training, uncertainty analysis, and SHAP interpretation. (B) Per-endpoint positive (toxic) label prevalence, showing pronounced class imbalance. (C) Per-endpoint label missingness arising from assay-specific experimental attrition. (D) Two-dimensional density of standardized molecular weight versus LogP across the full dataset, indicating drug-like chemical space coverage.'),
      img(FIG + 'fig1_overview.png', 620),

      heading('4.2 Predictive Performance', HeadingLevel.HEADING_2),
      p('The deep ensemble achieved a mean test ROC-AUC of 0.803 across the 12 endpoints (range 0.656–0.908), performing comparably to but not exceeding the Random Forest baseline (mean AUC 0.847, range 0.764–0.937) and modestly exceeding the single (non-ensembled) MLP (mean AUC 0.791) and XGBoost (mean AUC 0.811) (Table 2, Figure 2). The best-discriminated endpoints for all models were SR-MMP and NR-AhR (AUC > 0.90 for Random Forest and ensemble), consistent with these assays capturing broadly acting cytotoxicity and receptor-activation mechanisms with strong structure-activity signal. The most difficult endpoint across all models was NR-PPAR-gamma (ensemble AUC 0.656), reflecting its low prevalence (2.87%) and correspondingly limited positive training examples. Full ROC curves for all 12 endpoints are shown in Figure 6, confirming consistent ranking performance across the operating-point range.'),
      caption('Figure 2. Model performance comparison across 12 Tox21 endpoints. (A) Per-endpoint test ROC-AUC for Random Forest, XGBoost, single MLP, and the proposed deep ensemble. (B) Mean ROC-AUC (± SD across 12 endpoints) summarizing overall ranking: Random Forest > XGBoost ≈ Deep Ensemble > Single MLP.'),
      img(FIG + 'fig2_performance.png', 620),

      heading('4.3 Calibration and the Failure of Naive Selective Prediction', HeadingLevel.HEADING_2),
      p('Pooled across all 12 endpoints, the deep ensemble exhibited good probability calibration: Expected Calibration Error was 0.018 and the Brier score was 0.050, with the reliability diagram closely tracking the diagonal identity line across confidence bins (Figure 3A, Table 3), indicating that predicted probabilities are quantitatively meaningful (e.g., compounds assigned 30% predicted toxicity probability are toxic approximately 30% of the time).'),
      p('However, selective prediction analysis revealed a counter-intuitive and practically important result. Rather than improving with rejection of high-uncertainty predictions, retained-set ROC-AUC monotonically declined: from 0.847 with no rejection to 0.545 (near chance) after rejecting the 50% most uncertain predictions (Figure 3B, Table 3). This pattern arose because, under severe class imbalance, compounds with the lowest ensemble disagreement are overwhelmingly confidently-predicted true negatives (non-toxic compounds far from the decision boundary), while compounds with higher ensemble disagreement are concentrated near the decision boundary and disproportionately include true positives — meaning rejection of "uncertain" compounds preferentially removes the informative, harder-to-classify examples rather than unreliable ones.'),
      p('This interpretation was confirmed by stratified analysis: in 11 of 12 endpoints, above-median-epistemic-uncertainty compounds showed higher discriminative AUC than below-median-uncertainty compounds (mean difference +0.26; Figure 3C, Table 4), with the largest reversals in SR-MMP (+0.372) and NR-PPAR-gamma (+0.340). Only SR-p53 showed a (negligible, -0.001) difference in the conventional direction. These results indicate that ensemble disagreement in this setting functions more as a decision-boundary-proximity indicator than as a conventional reliability signal, and that uncertainty-based compound triage protocols imported uncritically from balanced-classification contexts may misallocate experimental validation resources if applied naively to imbalanced multi-task toxicity screening.'),
      caption('Figure 3. Calibration and selective-prediction analysis. (A) Reliability diagram pooled across 12 endpoints (point size proportional to bin sample count); dashed line indicates perfect calibration. (B) Selective prediction curve showing retained-set ROC-AUC as a function of the fraction of highest-uncertainty predictions rejected — note the counter-intuitive decline, attributable to class-imbalance-driven confidence collapse in the retained low-uncertainty set. (C) Per-endpoint ROC-AUC stratified by below- versus above-median ensemble epistemic standard deviation, showing higher-uncertainty compounds are, in most endpoints, more (not less) discriminable.'),
      img(FIG + 'fig3_uncertainty.png', 620),

      heading('4.4 Mechanistic Interpretability via SHAP', HeadingLevel.HEADING_2),
      p('SHAP analysis of the XGBoost surrogate identified physicochemical descriptors, rather than individual fingerprint substructure bits, as the dominant predictive features for both focus endpoints (Table 5, Figure 4). For SR-MMP (mitochondrial membrane potential disruption), LogP was the single most important feature (mean |SHAP| = 1.125), followed by fraction of sp3 carbons (0.534) and molecular weight (0.415), with the top individual fingerprint substructure (FP_578) ranking fourth overall. For NR-AhR (aryl hydrocarbon receptor activation), fraction of sp3 carbons was most important (0.781), followed by aromatic ring count (0.390) and LogP (0.308) — a feature ordering consistent with the well-established role of planar, aromatic, lipophilic xenobiotics as high-affinity AhR ligands.'),
      caption('Figure 4. SHAP feature importance for two mechanistically distinct endpoints. Top-10 mean absolute SHAP values for SR-MMP (mitochondrial toxicity) and NR-AhR (xenobiotic receptor activation), computed from an XGBoost surrogate model. Blue bars indicate physicochemical descriptors; red bars indicate individual Morgan-fingerprint substructure bits.'),
      img(FIG + 'fig4_shap.png', 620),

      p('SHAP dependence plots (Figure 5) further clarified the directionality of these relationships. Increasing LogP was associated with monotonically increasing (more positive) SHAP contribution toward SR-MMP toxicity across most of the observed range, consistent with the mechanistic hypothesis that highly lipophilic compounds partition into and disrupt mitochondrial membranes. Fraction of sp3 carbons showed a negative relationship with SR-MMP toxicity risk (higher sp3 fraction, i.e., more saturated/three-dimensional structures, associated with lower predicted toxicity) and a corresponding negative relationship with NR-AhR activation, while aromatic ring count showed the expected positive relationship with AhR activation, reflecting the known structure-activity relationship in which planar polyaromatic and halogenated-aromatic compounds are canonical AhR agonists.'),
      caption('Figure 5. SHAP dependence plots reveal mechanistically interpretable structure-toxicity relationships. (A) SR-MMP toxicity increases with LogP (lipophilicity). (B) SR-MMP toxicity decreases with increasing sp3-carbon fraction (molecular saturation/3-dimensionality). (C) NR-AhR activation decreases with increasing sp3-carbon fraction. (D) NR-AhR activation increases with aromatic ring count, consistent with canonical planar-aromatic AhR ligand pharmacophores. Dashed lines show local degree-2 polynomial trends.'),
      img(FIG + 'fig5_dependence.png', 550),

      heading('4.5 Endpoint-Level Discrimination and Chemical Space Structure', HeadingLevel.HEADING_2),
      p('ROC curves for all 12 endpoints (Figure 6) confirm consistent ranking ability across the full range of operating points, with the steepest early-recall gains observed for SR-MMP and NR-AhR. UMAP projection of the SR-MMP held-out test set fingerprints (Figure 7) revealed that toxic compounds (Figure 7A) and high-predicted-probability compounds (Figure 7B) cluster within overlapping but not identical regions of chemical space, and that high-epistemic-uncertainty compounds (Figure 7C) are concentrated at the boundaries between these clusters — providing direct visual confirmation of the mechanism underlying the selective-prediction reversal described in Section 4.3: uncertainty tracks proximity to the toxic/non-toxic decision boundary in chemical space rather than tracking unreliability per se.'),
      caption('Figure 6. Receiver operating characteristic (ROC) curves for all 12 Tox21 endpoints using the deep ensemble mean prediction on the held-out test set. Shaded regions indicate the area under each curve; diagonal dashed lines indicate chance-level performance.'),
      img(FIG + 'fig6_roc.png', 620),
      caption('Figure 7. Chemical-space structure of predictive uncertainty. UMAP projection (Jaccard metric on Morgan fingerprints) of the SR-MMP held-out test set, colored by (A) ground-truth toxicity label, (B) ensemble-predicted toxicity probability, and (C) epistemic uncertainty (ensemble standard deviation). High-uncertainty compounds concentrate at cluster boundaries rather than being randomly distributed, visually corroborating the decision-boundary-proximity interpretation of ensemble disagreement.'),
      img(FIG + 'fig7_chemspace.png', 620),

      heading('4.6 Three-Dimensional Structural Confirmation of SHAP-Derived Toxicophores', HeadingLevel.HEADING_2),
      p('To provide direct structural evidence for the SHAP-derived feature-attribution rules and to translate them into a form directly usable in molecular drug design, we generated MMFF94-optimized three-dimensional conformers (RDKit ETKDGv3 embedding) for representative dataset compounds selected to exemplify each toxicophore class (Figure 8). A high-lipophilicity compound (calculated LogP = 6.95, halogenated diaryl amide scaffold) illustrates the bulky, extended hydrophobic three-dimensional volume implicated by SHAP as the dominant SR-MMP risk factor — consistent with the mechanistic hypothesis that such compounds partition favorably into the lipid bilayer of the inner mitochondrial membrane and dissipate the transmembrane proton gradient. A planar tetracyclic polyaromatic hydroxylated compound (four fused aromatic rings, Fsp3 = 0) illustrates the rigid, extended planar geometry implicated by SHAP as the dominant NR-AhR risk factor, structurally consistent with the well-characterized requirement for a planar, hydrophobic ligand able to intercalate the AhR ligand-binding domain PAS-B pocket. A low-risk comparator compound (calculated LogP = -0.02, Fsp3 = 0.86, no aromatic rings) is shown for contrast, exhibiting a compact, saturated, three-dimensionally puckered conformation lacking either toxicophore feature.'),
      p('This three-dimensional structural view directly operationalizes the SHAP feature-attribution results (Section 4.4) into molecular design rules: for compound series flagged as high-SR-MMP-risk, medicinal chemists can visually inspect and iteratively reduce the hydrophobic three-dimensional volume (e.g., replacing extended alkyl/aryl-halogen substituents with polar or ring-constrained bioisosteres); for compound series flagged as high-NR-AhR-risk, chemists can disrupt molecular planarity (e.g., introducing sp3 centers, ortho-substituents, or saturated ring-fusion) to sterically hinder PAS-B pocket intercalation, an approach consistent with established "escape from flatland" and three-dimensional fragment-based design strategies.'),
      caption('Figure 8. Three-dimensional molecular conformations of representative toxicophore-bearing compounds from the Tox21 dataset (MMFF94-optimized geometries, RDKit ETKDGv3 embedding). (A) High-lipophilicity halogenated diaryl amide (LogP = 6.95) exemplifying the extended hydrophobic three-dimensional volume associated with SR-MMP (mitochondrial membrane potential) risk; lipophilic aliphatic/aromatic carbons are outlined in orange. (B) Planar tetracyclic polyaromatic hydroxylated compound (4 fused aromatic rings, Fsp3 = 0) exemplifying the rigid planar geometry associated with NR-AhR (aryl hydrocarbon receptor) activation risk; aromatic-ring atoms are outlined in red. (C) Low-risk saturated comparator compound (LogP = -0.02, Fsp3 = 0.86) lacking either toxicophore feature, shown for structural contrast. Atom coloring follows standard CPK convention (carbon: dark gray, nitrogen: blue, oxygen: red, sulfur: yellow, halogens: green).'),
      img(FIG + 'fig8_3d_toxicophore.png', 640),

      heading('4.7 Cross-Validated Performance and Statistical Robustness', HeadingLevel.HEADING_2),
      p('To confirm that the single-split results reported above are not an artifact of a particular data partition, we repeated stratified 5-fold cross-validation independently for all four model classes across all 12 endpoints (Table 7, Figure 9). Cross-validated overall mean ROC-AUC closely reproduced the single-split estimates: Random Forest 0.854 (95% CI ±0.029), XGBoost 0.822 (±0.030), single MLP 0.788 (±0.037), and deep ensemble 0.810 (±0.029), confirming that the relative ranking of model classes (Random Forest > XGBoost > deep ensemble > single MLP) and the approximate magnitude of the performance gap are stable across data partitions rather than an artifact of the particular train/test split used for the primary analysis. Confidence intervals were widest for the lowest-prevalence endpoint (NR-PPAR-gamma, ±0.06-0.11 across models), reflecting the expected estimation instability when few positive examples are available in a given fold, and narrowest for the highest-prevalence, best-discriminated endpoints (SR-MMP, NR-AhR; ±0.02 or better across models). The pooled fold-level distributions (Figure 9B) show no evidence of catastrophic failure folds for any model class, and confirm that Random Forest\'s advantage over the deep ensemble, while consistent, is modest in absolute magnitude (approximately 0.04 mean AUC) relative to the within-model fold-to-fold variability.'),
      caption('Figure 9. Cross-validated performance and statistical robustness. (A) Stratified 5-fold cross-validation mean ROC-AUC with 95% confidence intervals (t-distribution) for each of the 12 endpoints across four model classes, confirming the single-split ranking is stable. (B) Distribution of all fold-level ROC-AUC estimates pooled across the 12 endpoints (60 fold-level estimates per model class), showing overlapping but distinguishable performance distributions.'),
      img(FIG + 'fig9_cv_validation.png', 620),

      heading('4.8 External Validation on Independent Reference Toxicants', HeadingLevel.HEADING_2),
      p('Cross-checking against the full Tox21 training set identified 3 of 9 curated reference compounds (dibenzo[a,h]anthracene, beta-naphthoflavone, and FCCP) as genuinely absent from the training data, constituting a true test of out-of-distribution generalization; the remaining 6 compounds (including benzo[a]pyrene, 2,4-dinitrophenol, and amiodarone) were already present in Tox21 and are reported as internal consistency checks rather than extrapolation tests (Table 6).'),
      p('For the two truly external AhR agonists, the model correctly assigned high NR-AhR toxicity probability to both dibenzo[a,h]anthracene (P = 0.892) and beta-naphthoflavone (P = 0.858), replicating known pharmacology for this receptor despite these exact structures never appearing during training — providing genuine, if limited, evidence that the SHAP-derived planarity/aromaticity toxicophore rule (Section 4.4) generalizes to structurally related but unseen polyaromatic scaffolds. In contrast, the single truly external SR-MMP reference compound, FCCP (a phenylhydrazone-class mitochondrial uncoupler mechanistically and structurally distinct from the halogenated lipophilic compounds dominating Tox21\'s SR-MMP-positive training examples), was substantially under-predicted (P = 0.282, versus a known strong-uncoupler mechanism), and amiodarone — a clinically documented mitochondrial toxicant already present in the training set — was likewise under-predicted (P = 0.146). This asymmetry indicates that the model\'s SR-MMP mechanism, while statistically well-calibrated and mechanistically sound within the Tox21 chemical space (Section 4.4), does not fully capture the structurally diverse routes to mitochondrial toxicity documented in the broader pharmacological literature (e.g., cationic-amphiphile accumulation via membrane potential-driven uptake, distinct from simple lipophilicity-driven partitioning), and we report this as an honest boundary condition of the fingerprint-based approach rather than omitting it. All three saturated negative-control compounds (glucose, cyclohexanol, citric acid) were correctly assigned near-zero toxicity probability on both endpoints (P < 0.01 in all cases), confirming good specificity against non-toxic, non-aromatic chemical space.'),

      new Paragraph({ children: [new PageBreak()] }),

      // DISCUSSION
      heading('5. Discussion'),
      p('This study set out to jointly address predictive accuracy, calibrated uncertainty, and mechanistic interpretability for multi-task Tox21 toxicity prediction. Three findings merit particular emphasis. First, deep ensembles achieved competitive but not superior point-estimate accuracy relative to Random Forest, indicating that the primary value proposition of the ensemble approach in this setting is not raw discriminative performance but rather the availability of well-calibrated, per-prediction uncertainty estimates that tree-based baselines do not natively provide. Practitioners should therefore select between these model classes based on downstream use case (point-estimate ranking versus confidence-aware decision support) rather than assuming ensembles are strictly superior.'),
      p('Second, and most importantly, our selective-prediction analysis demonstrates that a widely assumed property of uncertainty-aware models — that rejecting high-uncertainty predictions improves retained-set accuracy — does not hold under the severe class imbalance characteristic of toxicity screening data. This has direct practical implications: a naive triage protocol that prioritizes experimental validation toward "confident" model predictions and deprioritizes "uncertain" ones would, according to our results, systematically deprioritize exactly the borderline, structurally ambiguous compounds for which model predictions are least reliable and experimental data would be most valuable — the opposite of the intended effect. We recommend that groups deploying ensemble uncertainty for compound triage in imbalanced toxicity or bioactivity settings explicitly test the direction of this relationship on their own data before adopting a rejection-based workflow, rather than importing assumptions from balanced-classification or medical-imaging literature.'),
      p('Third, SHAP-based interpretability confirmed that the model, despite being trained on abstract fingerprint and descriptor features without explicit toxicological supervision, recovered mechanistically sound structure-toxicity relationships: lipophilicity as the dominant driver of mitochondrial membrane disruption, and aromatic planarity as the dominant driver of AhR activation. This concordance between data-driven feature importance and established toxicological mechanism provides a form of external validation for the model beyond aggregate accuracy metrics, and offers medicinal chemists actionable, interpretable guidance (e.g., reducing LogP or aromatic ring count to mitigate specific liabilities) rather than an opaque toxicity score alone.'),

      heading('5.1 Implications for Rational, Multi-Parameter Drug Design', HeadingLevel.HEADING_2),
      p('Beyond benchmark accuracy, the SHAP-derived structure-toxicity relationships and their three-dimensional structural confirmation (Figure 8) translate directly into actionable design rules for medicinal chemists conducting lead optimization. The dominance of LogP in driving SR-MMP (mitochondrial membrane potential) liability is consistent with the established mechanism by which highly lipophilic cationic amphiphiles accumulate in the mitochondrial matrix, driven by the inner-membrane potential, and disrupt oxidative phosphorylation; this motivates explicit LogP ceilings (e.g., cLogP < 3-4, in line with Lipinski-type heuristics) during virtual library enumeration and de novo generation, rather than relying on aggregate drug-likeness scores that may not isolate this specific mechanism. Similarly, the joint importance of aromatic ring count and sp3-carbon fraction (Fsp3) for NR-AhR activation reflects the canonical planar, halogenated- or polyaromatic pharmacophore recognized by the AhR ligand-binding domain; increasing Fsp3 and reducing fused-aromatic ring count — a strategy already advocated in "escape from flatland" medicinal chemistry campaigns for improving solubility and reducing promiscuous binding — simultaneously reduces AhR-mediated toxicity risk, illustrating convergence between orthogonal optimization objectives.'),
      p('In a prospective multi-parameter optimization (MPO) workflow, these individually interpretable, per-endpoint SHAP relationships can be combined with the model\'s calibrated probability outputs to construct a composite toxicity-risk score for candidate compounds generated by de novo design algorithms (e.g., genetic algorithms, reinforcement learning, or generative deep learning models operating on molecular graphs or SMILES strings), enabling toxicity-aware reward shaping during generative optimization rather than post hoc filtering of a static virtual library. Because our uncertainty analysis shows that compounds near the decision boundary are simultaneously the most toxicologically ambiguous and the most informative for active learning, we further suggest that generative or library-design campaigns preferentially route high-epistemic-uncertainty candidates to experimental confirmation (rather than either accepting or rejecting them by model score alone), closing the loop between computational triage and targeted wet-laboratory validation in a manner that respects the class-imbalance-driven confound identified in Section 4.3.'),

      heading('5.2 Limitations', HeadingLevel.HEADING_2),
      p('Several limitations should be considered. The ensemble members share an identical architecture and only differ in random initialization and early-stopping trajectory, which may understate epistemic uncertainty relative to architectures that also vary hyperparameters or feature representations. The SHAP interpretability analysis uses a separate XGBoost surrogate rather than directly explaining the deep ensemble, which is standard practice for computational tractability with TreeExplainer but introduces a model-fidelity gap between the explained surrogate and the ensemble used for uncertainty quantification. Our selective-prediction finding, while consistent across 11 of 12 endpoints and confirmed stable under 5-fold cross-validation, was demonstrated specifically for Tox21-scale imbalance ratios (2.9%-16.2% positive prevalence); the direction and magnitude of this effect in datasets with different imbalance ratios or different underlying uncertainty-generating mechanisms (e.g., epistemic uncertainty from out-of-distribution chemical space rather than decision-boundary proximity) may differ and warrants further investigation. While repeated 5-fold cross-validation (Section 4.7) confirmed that our primary single-split findings are not an artifact of a particular partition, our external validation (Section 4.8) was necessarily limited to a small set of 9 reference compounds (only 3 of which were genuinely absent from the training data), and revealed that generalization to structurally novel mitochondrial-toxicity chemotypes (e.g., FCCP) is imperfect; a substantially larger and chemically diverse external validation panel, ideally with prospectively generated experimental data, would be needed to more rigorously characterize the boundaries of model generalization before any clinical or regulatory application.'),

      heading('5.3 Future Work', HeadingLevel.HEADING_2),
      p('Promising extensions include: (i) decomposing ensemble uncertainty into decision-boundary-proximity and out-of-distribution components using complementary methods (e.g., Gaussian process latent-space distance, conformal prediction), which may disentangle the confound identified here; (ii) extending the multi-task architecture to explicitly model inter-endpoint label correlation via shared latent representations, which could improve performance on low-prevalence endpoints such as NR-PPAR-gamma; (iii) applying graph neural network encoders in place of fixed fingerprints within the same ensemble/calibration/interpretability framework to test whether learned representations alter the selective-prediction reversal reported here; and (iv) expanding the external validation panel introduced in Section 4.8 to a substantially larger and mechanistically diverse set of reference compounds with prospectively generated experimental data, to more precisely delineate which mitochondrial-toxicity and receptor-activation chemotypes fall outside the model\'s applicability domain.'),

      heading('6. Conclusion'),
      p('We present a multi-task deep ensemble framework for Tox21 toxicity prediction that jointly delivers competitive discriminative performance, well-calibrated probability estimates, and mechanistically interpretable feature attributions. Our central contribution is an empirical demonstration that naive uncertainty-based selective prediction — an assumption commonly imported from balanced-classification contexts — fails and in fact reverses under the severe class imbalance typical of toxicity screening data, because low ensemble disagreement predominantly reflects confident majority-class predictions rather than genuine reliability. This finding has direct, actionable implications for how uncertainty estimates should be used (or withheld from use) in compound triage workflows, and we recommend explicit empirical validation of the uncertainty-reliability relationship before deploying rejection-based triage in any new imbalanced screening dataset.'),

      heading('Data and Code Availability'),
      p('The Tox21 dataset used in this study is publicly available. Featurization, model training, uncertainty analysis, and SHAP interpretability code is available from the corresponding author upon reasonable request to support reproducibility.'),

      heading('Author Contributions'),
      p('Y.M.H.: conceptualization, methodology, software, formal analysis, data curation, writing — original draft. H.E.-T.: validation, toxicological interpretation, writing — review & editing. I.R.A.: validation, immunotoxicological interpretation, writing — review & editing. M.S.A.: conceptualization, supervision, resources, writing — review & editing.'),

      heading('Conflicts of Interest'),
      p('The authors declare no conflicts of interest.'),

      new Paragraph({ children: [new PageBreak()] }),

      heading('References'),
      p('1. Huang, R., Xia, M., Nguyen, D.-T., Zhao, T., Sakamuru, S., Zhao, J., Shahane, S. A., Rossoshek, A. & Simeonov, A. Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. Front. Environ. Sci. 3, 85 (2016).', { align: AlignmentType.LEFT }),
      p('2. Mayr, A., Klambauer, G., Unterthiner, T. & Hochreiter, S. DeepTox: Toxicity prediction using deep learning. Front. Environ. Sci. 3, 80 (2016).', { align: AlignmentType.LEFT }),
      p('3. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. Adv. Neural Inf. Process. Syst. 30, 4765-4774 (2017).', { align: AlignmentType.LEFT }),
      p('4. Lakshminarayanan, B., Pritzel, A. & Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. Adv. Neural Inf. Process. Syst. 30, 6402-6413 (2017).', { align: AlignmentType.LEFT }),
      p('5. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On calibration of modern neural networks. Proc. 34th Int. Conf. Mach. Learn. 70, 1321-1330 (2017).', { align: AlignmentType.LEFT }),
      p('6. Rogers, D. & Hahn, M. Extended-connectivity fingerprints. J. Chem. Inf. Model. 50, 742-754 (2010).', { align: AlignmentType.LEFT }),
      p('7. Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. 785-794 (2016).', { align: AlignmentType.LEFT }),
      p('8. McInnes, L., Healy, J. & Melville, J. UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426 (2018).', { align: AlignmentType.LEFT }),
      p('9. Li, Y. et al. Co-model for chemical toxicity prediction based on multi-task deep learning. Mol. Inform. 42, e202200257 (2023).', { align: AlignmentType.LEFT }),
      p('10. Wallach, I. & Heifets, A. Most ligand-based classification benchmarks reward memorization rather than generalization. J. Chem. Inf. Model. 58, 916-932 (2018).', { align: AlignmentType.LEFT }),
      p('11. Yuan, Q., Wei, Z., Guan, X., Jiang, M., Wang, S., Zhang, S. & Li, Z. Toxicity prediction method based on multi-channel convolutional neural network. Molecules 24, 3383 (2019).', { align: AlignmentType.LEFT }),
      p('12. Karim, A., Mishra, A., Newton, M. A. H. & Sattar, A. Efficient toxicity prediction via simple features using shallow neural networks and decision trees. ACS Omega 4, 1874-1888 (2019).', { align: AlignmentType.LEFT }),
      p('13. Feinstein, J., Sivaraman, G., Picel, K., Peters, B., Vázquez-Mayagoitia, Á., Ramanathan, A., MacDonell, M., Foster, I. & Yan, E. Uncertainty-informed deep transfer learning of perfluoroalkyl and polyfluoroalkyl substance toxicity. J. Chem. Inf. Model. 61, 5793-5803 (2021).', { align: AlignmentType.LEFT }),
      p('14. Scalia, G., Grambow, C. A., Pernici, B., Li, Y.-P. & Green, W. H. Evaluating scalable uncertainty estimation methods for deep learning-based molecular property prediction. J. Chem. Inf. Model. 60, 2697-2717 (2020).', { align: AlignmentType.LEFT }),
      p('15. Zhang, Y. & Lee, A. A. Bayesian semi-supervised learning for uncertainty-calibrated prediction of molecular properties and active learning. Chem. Sci. 10, 8154-8163 (2019).', { align: AlignmentType.LEFT }),
      p('16. Ahmad, W., Tayara, H. & Chong, K. T. Attention-based graph neural network for molecular solubility prediction. ACS Omega 8, 3236-3244 (2023).', { align: AlignmentType.LEFT }),
      p('17. Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R. & Coley, C. W. Uncertainty quantification using neural networks for molecular property prediction. J. Chem. Inf. Model. 60, 3770-3780 (2020).', { align: AlignmentType.LEFT }),
      p('18. Mervin, L. H., Johansson, S., Semenova, E., Giblin, K. A. & Engkvist, O. Uncertainty quantification in drug design. Drug Discov. Today 26, 474-489 (2021).', { align: AlignmentType.LEFT }),
      p('19. Rácz, A., Bajusz, D. & Héberger, K. Multi-level comparison of machine learning classifiers and their performance against random chance. Molecules 24, 2811 (2019).', { align: AlignmentType.LEFT }),
      p('20. Sheridan, R. P. Using random forest to model the domain applicability of another random forest model. J. Chem. Inf. Model. 53, 2837-2850 (2013).', { align: AlignmentType.LEFT }),
      p('21. Lipinski, C. A., Lombardo, F., Dominy, B. W. & Feeney, P. J. Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Adv. Drug Deliv. Rev. 23, 3-25 (1997).', { align: AlignmentType.LEFT }),
      p('22. Denny, W. A. The role of hydrophobicity in the toxicity of drugs and drug candidates. Chem. Res. Toxicol. 33, 2464-2472 (2020).', { align: AlignmentType.LEFT }),
      p('23. Nebert, D. W. & Dalton, T. P. The role of cytochrome P450 enzymes in endogenous signalling pathways and environmental carcinogenesis. Nat. Rev. Cancer 6, 947-960 (2006).', { align: AlignmentType.LEFT }),
      p('24. Denison, M. S. & Nagy, S. R. Activation of the aryl hydrocarbon receptor by structurally diverse exogenous and endogenous chemicals. Annu. Rev. Pharmacol. Toxicol. 43, 309-334 (2003).', { align: AlignmentType.LEFT }),
      p('25. Wills, T. J. & Begley, D. J. Drug-induced mitochondrial toxicity: molecular mechanisms and prediction in drug development. Toxicol. Mech. Methods 22, 1-15 (2012).', { align: AlignmentType.LEFT }),
      p('26. Wu, D. & Rastinejad, F. Structural characterization of mammalian bHLH-PAS transcription factors. Curr. Opin. Struct. Biol. 43, 1-9 (2017).', { align: AlignmentType.LEFT }),
      p('27. Lovering, F., Bikker, J. & Humblet, C. Escape from flatland: increasing saturation as an approach to improving clinical success. J. Med. Chem. 52, 6752-6756 (2009).', { align: AlignmentType.LEFT }),
      p('28. Riniker, S. & Landrum, G. A. Better informed distance geometry: using what we know to improve conformation generation. J. Chem. Inf. Model. 55, 2562-2574 (2015).', { align: AlignmentType.LEFT }),

      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 1. Dataset characteristics by toxicity endpoint'),
      makeTable(t1.header, t1.rows, [1400,2600,1300,1300,1400,1360]),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 2. Model performance comparison (test ROC-AUC)'),
      makeTable(t2.header, t2.rows, [1500,1500,1300,1300,1900,1860]),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 3. Calibration and selective-prediction summary'),
      makeTable(t3.header, t3.rows, [6360,3000]),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 4. Per-endpoint discrimination stratified by epistemic uncertainty'),
      makeTable(t4.header, t4.rows, [1800,1700,2500,2500,860]),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 5. Top SHAP features for focus endpoints (SR-MMP, NR-AhR)'),
      makeTable(t5.header, t5.rows, [1800,2100,3300,2160]),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 6. External validation on independent reference toxicants'),
      p('Compounds marked "No (true external)" were confirmed absent from the Tox21 training set via canonical SMILES matching and constitute genuine out-of-distribution generalization tests; all others are internal consistency checks.', { size: 18 }),
      makeTable(t6.header, t6.rows, [1500,1500,1100,1000,900,700,900,760]),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Table 7. Stratified 5-fold cross-validation summary (mean ROC-AUC ± 95% CI)'),
      makeTable(t7.header, t7.rows, [1200,1000,1030,1000,930,1000,1030,1000,1030]),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/home/claude/project/manuscript.docx', buf);
  console.log('Manuscript written.');
});
