# Training Caveats and Interpretation Guide

This document explains the constraints of the training run in Section 3 and what they mean for interpreting the evaluation results in this section.

## 1. Conservative Learning Rate

**What we used:** `5e-6` (0.000005)

**What is typical:** LoRA fine-tuning commonly uses learning rates between `1e-4` and `2e-4` (20x to 40x higher).

**Impact:** A low learning rate means the model updates its weights very slowly per training step. With only 8 training examples over 5 epochs, the model had limited opportunity to shift its internal representations. This is a deliberate trade-off: lower learning rates reduce the risk of catastrophic forgetting (the model losing its general capabilities) but also reduce how much new knowledge is absorbed.

**In production:** You would typically start with `2e-4` for LoRA and reduce only if you observe instability or degradation on general benchmarks.

## 2. Loss Plateau

**What we observed:** Training loss decreased from 2.93 to 2.26 over 5 epochs.

**What good convergence looks like:** A prior run with standard hyperparameters showed loss decreasing from 2.84 to 0.55. A loss below 1.0 generally indicates the model has learned the training examples well.

**Impact:** A final loss of 2.26 means the model is still frequently "surprised" by the correct answers in the training data. It has partially learned the patterns but has not memorized or fully internalized them. This is consistent with the conservative learning rate — the model simply did not update enough to fully learn the training distribution.

**In production:** You would train until loss plateaus below 1.0, potentially for 10-20 epochs with a higher learning rate. You would also monitor validation loss to avoid overfitting.

## 3. Training Data Scope

**What we used:** 8 training examples, all about Thief abilities from the Basic Fantasy RPG rules.

**What the evaluation covers:** 10 questions spanning Thieves, Elves, Fighters, Clerics, Magic-Users, Halflings, retainers, and hirelings.

**Impact:** The training data covers at most 1-2 of the 10 evaluation questions (those related to Thief mechanics). The remaining 8-9 questions test topics the model was never trained on. This means:
- Improvement on Thief questions = training worked for covered topics
- No improvement on non-Thief questions = expected outcome, not a failure
- Improvement on non-Thief questions = surprising generalization (unlikely with 8 examples)

**In production:** You would generate training data covering the full document corpus. Section 2's SDG pipeline can produce hundreds of QA pairs across all topics. A diverse training set is essential for broad domain adaptation.

## 4. What Would Be Different in Production

| Aspect | Lab Setting | Production Setting |
|--------|------------|-------------------|
| Training examples | 8 (single topic) | 500-2000+ (full domain) |
| Learning rate | 5e-6 | 1e-4 to 2e-4 |
| Epochs | 5 | 10-20 (until convergence) |
| Final loss | ~2.26 | < 1.0 target |
| Evaluation | Same 10 questions | Held-out test set (50-100+) |
| Training data coverage | Thief abilities only | All document topics |
| Quality filtering | Faithfulness check only | Faithfulness + diversity + dedup |

## 5. Key Takeaway

The lab training run is a **process demonstration**, not a performance benchmark. The value is in the pipeline:

1. **SDG** generates training data from customer documents
2. **Training** adapts the model using that data
3. **Evaluation** measures whether the adaptation worked

Each stage has clear inputs, outputs, and knobs to tune. In a customer engagement, you adjust the knobs (more data, higher learning rate, more epochs) to hit the performance targets. The pipeline itself does not change.
