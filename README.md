# dr-early-detection
Early detection of diabetic retinopathy from retinal fundus images (course project)
# Early Detection of Diabetic Retinopathy from Retinal Fundus Images

Course project for **Research Methods and Tools** (Astana IT University).

**Authors:** Tamam Alhendi, Sogdiana Marajapova

> **Status:** the study design is complete and implementation is in progress.
> No trained models or experimental results are available yet.

## Overview

This project develops an intelligent system for the **early detection of diabetic
retinopathy (DR)**, meaning any DR (ICDR grade ≥ 1) versus no DR, from colour
fundus photographs. The system combines:

- a **quality gate** (ResNet-50 trained on EyeQ labels) that rejects unreadable images;
- the **RETFound** retinal foundation model (ViT-Large) as the feature extractor;
- a **lesion-aware attention head** that produces intrinsic lesion heatmaps
  from image-level labels only;
- a **focal multi-task loss** that handles class imbalance, so that grade 1
  (mild NPDR) is detected more often.

## Research questions

1. **RQ1:** Does the proposed system detect any DR, and grade 1 in particular,
   more accurately than an ImageNet CNN and a standard fine-tuned RETFound?
2. **RQ2:** Does performance hold on external data from other countries and cameras?
3. **RQ3:** Do the built-in attention maps locate real lesions more often than Grad-CAM?
4. **RQ4:** How much do the quality gate, the loss and the input resolution each contribute?

## Experimental conditions

| ID | Model | Purpose |
|----|-------|---------|
| B1 | ResNet-50 (ImageNet), cross-entropy | CNN baseline |
| B2 | RETFound + linear head | Foundation-model baseline |
| P  | RETFound + lesion-aware attention + focal loss + quality gate | Proposed system |
| A1 | P without quality gate | Ablation |
| A2 | P with plain cross-entropy | Ablation |
| A3 | P at 448×448 instead of 224×224 | Ablation |

Each condition is trained with three random seeds (0, 1, 2).

## Datasets (public, not included in this repository)

| Dataset | Origin | Role |
|---------|--------|------|
| EyePACS (Kaggle 2015) | USA | Development |
| APTOS 2019 | India | Development |
| EyeQ | USA | Quality gate |
| Messidor-2 (adjudicated grades) | France | External test |
| IDRiD | India | External test, lesion masks |
| DDR | China | External test, lesion masks |

Download each dataset from its official source under its own licence.

## Evaluation

Sensitivity, specificity, PPV/NPV, AUC, quadratic weighted kappa, per-grade
recall and macro-F1, with 95% bootstrap confidence intervals (patient-level).
DeLong and McNemar tests are used for model comparisons. Lesion localisation
is measured with the pointing game.

## Planned structure
- `configs/` — experiment configurations
- `data/` — split definitions (no images)
- `src/` — preprocessing, models, training, evaluation
- `scripts/` — scripts that reproduce the tables and figures


## Planned environment

Python 3.10, PyTorch 2.x, timm, the official RETFound code and weights, OpenCV,
Albumentations, scikit-learn, SciPy and pytorch-grad-cam. Exact versions will be
pinned in `requirements.txt`.

## License

MIT
