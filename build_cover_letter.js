const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, AlignmentType, convertInchesToTwip } = require('docx');

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 200, ...(opts.spacing || {}) },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: Array.isArray(text) ? text : [new TextRun({ text, size: 22, italics: !!opts.italics, bold: !!opts.bold })],
  });
}

const doc = new Document({
  styles: { default: { document: { run: { font: 'Times New Roman', size: 22 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: [
      p('Youssef M. Hassan', { bold: true, spacing: { after: 40 } }),
      p('Department of Zoology, Faculty of Science, Ain Shams University', { spacing: { after: 40 } }),
      p('Abbassia 11566, Cairo, Egypt', { spacing: { after: 40 } }),
      p('Email: yousefmohamed_p@sci.asu.edu.eg', { spacing: { after: 300 } }),

      p('Date: [Insert submission date]', { spacing: { after: 300 } }),

      p('The Editor-in-Chief', { spacing: { after: 40 } }),
      p('Chemico-Biological Interactions', { spacing: { after: 40 } }),
      p('Elsevier', { spacing: { after: 300 } }),

      p('Re: Submission of original research article', { bold: true, spacing: { after: 200 } }),

      p('Dear Editor,', { spacing: { after: 200 } }),

      p('We are pleased to submit our original research article entitled "Development of a Machine Learning Framework with Calibrated Uncertainty Quantification and 3D Toxicophore Mapping for Multi-Task Prediction of Tox21 Chemical-Biological Interactions: Mechanistic Insights for Rational Drug Design" for consideration as an original research article in Chemico-Biological Interactions.'),

      p('This work directly addresses a mechanistic question at the core of your journal\'s scope: what molecular and structural features govern the chemical-biological interactions underlying xenobiotic toxicity, and how reliably can these interactions be predicted computationally? Rather than presenting another incremental benchmark exercise, our central contribution is an empirical, mechanistically-grounded finding with direct practical consequences for how computational toxicology is used to support drug safety assessment.'),

      p('Specifically, we show that a widely assumed property of uncertainty-aware machine learning models — that rejecting the model\'s most uncertain predictions improves the reliability of the predictions that remain — does not hold, and in fact reverses, under the severe class imbalance characteristic of nuclear receptor and stress-response toxicity screening data (as in the Tox21 panel). We demonstrate that this occurs because low model disagreement in this setting predominantly reflects confident majority-class (non-toxic) predictions rather than genuine reliability, and we confirm the finding is stable under repeated 5-fold cross-validation across all 12 endpoints. This has direct, actionable implications for toxicologists and computational chemists who might otherwise deprioritize "uncertain" compounds during virtual screening triage — our results indicate this would systematically discard the most chemically ambiguous and experimentally informative compounds.'),

      p('We complement this uncertainty analysis with SHapley Additive exPlanation (SHAP)-based mechanistic interpretation, which recovers toxicologically sound structure-activity relationships without any explicit mechanistic supervision: lipophilicity (LogP) as the dominant driver of mitochondrial membrane potential disruption (SR-MMP), and molecular planarity/aromaticity as the dominant driver of aryl hydrocarbon receptor activation (NR-AhR) — both consistent with established chemical-biological interaction mechanisms in the toxicology literature. We further translate these findings into three-dimensional molecular structure, presenting MMFF94-optimized conformers of representative toxicophore-bearing compounds that visually and mechanistically ground the model\'s feature attributions, and we discuss direct implications for rational, structure-guided lead optimization.'),

      p('Finally, in the interest of scientific rigor, we subjected our model to external validation against nine well-characterized reference toxicants from the pharmacological literature, explicitly distinguishing compounds absent from the Tox21 training data (true generalization tests) from those already present (internal consistency checks). We report this analysis transparently, including a case in which the model under-predicted the toxicity of a structurally novel mitochondrial uncoupler (FCCP) — an honest boundary condition that we believe strengthens rather than weakens the paper\'s contribution to the field\'s understanding of computational toxicology\'s current limits.'),

      p('We believe this manuscript will be of interest to your readership because it moves beyond point-estimate accuracy benchmarking to directly interrogate the reliability, mechanistic validity, and generalization boundaries of machine learning models for chemical-biological toxicity prediction — questions of immediate practical relevance to toxicologists, pharmacologists, and medicinal chemists engaged in drug safety assessment.'),

      p('This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the manuscript and agree with its submission to Chemico-Biological Interactions. The authors declare no conflicts of interest.'),

      p('We suggest the following as potential reviewers given their expertise in computational toxicology and chemical-biological interaction mechanisms: [Insert 3-4 suggested reviewer names/emails/affiliations prior to submission].'),

      p('Thank you for considering our manuscript. We look forward to your response.'),

      p('Sincerely,', { spacing: { after: 300 } }),

      p('Youssef M. Hassan (corresponding author)', { spacing: { after: 20 } }),
      p('on behalf of all authors:', { spacing: { after: 20 } }),
      p('Hala El-Tantawi, Ibrahim Rabie Ali, Mohamed S. Attia', { spacing: { after: 20 } }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/home/claude/project/cover_letter.docx', buf);
  console.log('Cover letter written.');
});
