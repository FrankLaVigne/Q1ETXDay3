# Q1ETXDay3

Day 3 lab materials for the Red Hat AI Innovation hands-on workshop. This session covers the full path from synthetic data generation through model fine-tuning, using open-source tooling on OpenShift AI with GPU acceleration.

## Project Structure

```
Q1ETXDay3/
├── 01SDG/                          # Section 1: Synthetic Data Generation
│   ├── 01SDGHub.ipynb              # SDG Hub pipeline notebook
│   ├── Basic-Fantasy-RPG-Rules-r142.md  # Source document for SDG
│   └── synthetic_qa_pairs.csv      # Generated Q&A training pairs
├── 02/                             # Section 2: Model Download & Fine-Tuning
│   ├── 01.ipynb                    # Model setup and training notebook
│   └── models/                     # Downloaded model weights (git-ignored)
├── OSFT/                           # Orthogonal Subspace Fine-Tuning
│   └── OSFT_Interactive_Notebook.ipynb  # Interactive OFT/OSFT explainer
├── .gitignore
└── README.md
```

## Sections

### 1 - Synthetic Data Generation (SDG Hub)

Uses [SDG Hub](https://github.com/red-hat-ai-innovation/sdg-hub) to transform ingested documents into structured question-answer pairs suitable for fine-tuning. The notebook walks through flow discovery, model configuration, schema setup, seed dataset construction, dry runs, generation, iteration, and export.

### 2 - Model Download & Fine-Tuning

Downloads IBM Granite 3.2 8B Instruct from Hugging Face and prepares it for fine-tuning with [Training Hub](https://github.com/red-hat-ai-innovation/training-hub). Covers environment setup including flash-attn and CUDA dependencies on an NVIDIA L40S GPU.

### OSFT - Orthogonal Subspace Fine-Tuning

An interactive notebook exploring OFT and OSFT concepts with visualizations: orthogonal matrix properties, hyperspherical energy preservation, SVD decomposition, gradient projection, and parameter efficiency comparisons against LoRA and full fine-tuning.

## Environment

- Python 3.12
- PyTorch 2.7 + CUDA 12.8
- NVIDIA L40S (46 GB VRAM)
- OpenShift AI (RHODS)

## Key Dependencies

- `sdg-hub[examples]` - Synthetic data generation pipelines
- `training-hub[cuda]` - Model fine-tuning framework
- `flash-attn` - Flash Attention for efficient training
- `transformers`, `datasets` - Hugging Face ecosystem
