# AI Continuum Series - Model Adaptation Lab

Day 3 lab materials for the AI Continuum hands-on workshop series. This session walks through the full model-adaptation lifecycle: review Day 2 evidence, attempt inference-time scaling, generate synthetic training data, fine-tune with QLoRA, and evaluate the results — all on OpenShift AI with GPU acceleration.

## Project Structure

```
Q1ETXDay3/
├── 00Setup/                        # Section 0: Recap & Evidence
│   └── 00_Recap_and_Evidence.ipynb
├── 01Inference_Time_Scaling/       # Section 1: Inference-Time Scaling
│   └── 01_BestOfN.ipynb
├── 02SyntheticDataGen/             # Section 2: Synthetic Data Generation
│   ├── 01SDGHub.ipynb
│   └── Basic-Fantasy-RPG-Rules-r142.md
├── 03ModelAdaptation/              # Section 3: Model Fine-Tuning (QLoRA)
│   └── 01ModelAdaptation-Succeeded.ipynb
├── 04Evaluation/                   # Section 4: Evaluation
│   ├── 01_Evaluation.ipynb
│   └── 02_MultiModelEval.ipynb
├── 05Conclusion/                   # Section 5: Conclusion & Discussion
│   └── 01_Synthesis.ipynb
├── prebuilt/                       # Pre-generated results for offline use
│   ├── bon_results.json
│   ├── eval_results.json
│   ├── eval_with_context.json
│   ├── sdg_run1_results.csv
│   ├── sdg_run2_results.csv
│   └── synthetic_qa_pairs.csv
├── config.py
├── .gitignore
└── README.md
```

## Sections

### 0 - Recap and Evidence

Reconnects participants to the Day 2 workflow, loads the evaluation results from that session, and builds the evidence-based case for why RAG alone was insufficient and model adaptation is the next step.

### 1 - Inference-Time Scaling (Best-of-N)

Before committing to training, tests whether spending more compute at inference time can close the gap. Implements Best-of-N sampling to generate multiple candidate answers and select the best one, establishing a performance ceiling for the base model.

### 2 - Synthetic Data Generation (SDG Hub)

Uses [SDG Hub](https://github.com/red-hat-ai-innovation/sdg-hub) to transform the source document into structured question-answer pairs suitable for fine-tuning. Covers flow discovery, model configuration, schema setup, seed dataset construction, dry runs, generation, iteration, and export.

### 3 - Model Adaptation (QLoRA Fine-Tuning)

Fine-tunes IBM Granite 3.2 8B Instruct using QLoRA (4-bit quantized LoRA) with the synthetic data generated in Section 2. Includes the full training pipeline plus a quick inference check against the hardest evaluation questions.

### 4 - Evaluation

Loads the base model alongside the trained LoRA adapter and evaluates whether model adaptation closed the performance gap identified in Section 0. `01_Evaluation.ipynb` tests the lab-trained Granite 8B adapter. `02_MultiModelEval.ipynb` compares four fine-tuned models (Granite 8B, Granite 2B, Phi-3 Mini, Qwen2.5 3B) across architectures and sizes to answer whether architecture matters when training data is identical.

### 5 - Conclusion

A facilitated discussion (no code) that translates the lab results into customer-facing language. Covers the three questions customers actually ask, the escalation ladder as a trust-building tool, honest interpretation of the evaluation results, and what to bring into the next customer meeting.

## Environment

- Python 3.12
- PyTorch 2.7 + CUDA 12.8
- NVIDIA L40S (46 GB) or L4 (24 GB)
- OpenShift AI (RHODS)

## Key Dependencies

- `sdg-hub` - Synthetic data generation pipelines
- `unsloth` - Efficient QLoRA fine-tuning
- `peft`, `bitsandbytes` - Parameter-efficient fine-tuning and quantization
- `transformers`, `datasets`, `accelerate` - Hugging Face ecosystem
