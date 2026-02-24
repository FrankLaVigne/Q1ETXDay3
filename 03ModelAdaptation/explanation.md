# Troubleshooting: `selective_scan_cuda` ImportError in LoRA Training

## The Error

When running the LoRA SFT training cell in `modeltraining21.ipynb`, the following error was raised:

```
ImportError: Failed to import dependencies for Unsloth backend:
/opt/app-root/lib64/python3.12/site-packages/selective_scan_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

Install LoRA dependencies with: pip install 'training-hub[lora]'
```

Re-running `pip install 'training-hub[lora]'` (as the message suggests) does not fix the problem. The error is not about missing packages — it is about binary incompatibility between pre-compiled CUDA extensions and the installed PyTorch version.

## Root Cause Analysis

### Step 1: Identify the broken component

The error points to a specific shared library file:

```
selective_scan_cuda.cpython-312-x86_64-linux-gnu.so
```

This is a compiled CUDA extension from the `mamba-ssm` package (v2.3.0). The "undefined symbol" error means the `.so` file was compiled against a different version of PyTorch's C++ library (`libc10`) than what is currently installed.

The environment has **PyTorch 2.10.0+cu128**. The `mamba-ssm` package was pre-compiled against an older PyTorch version. When the dynamic linker tries to resolve the symbol `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib` (a demangled C++ function in `c10::cuda`), it cannot find it in the current PyTorch's `libc10.so`, and the import fails.

### Step 2: Trace the import chain

The error originates from cell 26 calling `lora_sft(...)`, which is a convenience function from `training_hub`. Here is the full import chain that triggers the failure:

```
lora_sft()                                          # notebook cell 26
  └─ training_hub/algorithms/lora.py line 17
     └─ from unsloth import FastLanguageModel        # triggers unsloth init
        └─ unsloth/__init__.py
           └─ from .models import *                  # loads models subpackage
              └─ unsloth/models/__init__.py
                 └─ from .loader import FastLanguageModel
                    └─ (module-level initialization in unsloth_zoo)
                       └─ import mamba_ssm            # tries to load mamba-ssm
                          └─ mamba_ssm loads selective_scan_cuda.so
                             └─ FAILS: undefined symbol
```

The `training_hub` wraps this with a `try/except ImportError` block (lines 16-37 in `lora.py`). Since the error message doesn't contain "unsloth" or "trl", it falls to the generic `else` branch, which re-raises with the "Failed to import dependencies for Unsloth backend" prefix.

### Step 3: Determine why mamba-ssm is being loaded

`mamba-ssm` provides CUDA kernels for **Mamba-architecture models** (state space models). It is an optional dependency of Unsloth that supports models like Falcon Mamba and Jamba. Unsloth tries to import it during initialization to patch Mamba-related operations.

**Granite 3.2 8B is a transformer model, not a Mamba model.** The `mamba-ssm` package is completely unnecessary for this training task.

### Step 4: Evaluate fix options

| Option | Pros | Cons |
|--------|------|------|
| Recompile `mamba-ssm` from source | Fixes properly | Takes 10-20 min, may fail with PyTorch 2.10 |
| Remove the broken `.so` file | Quick | Leaves a broken package installed |
| Uninstall `mamba-ssm` entirely | Clean, removes the problem at the source | Must reinstall if Mamba models are ever needed |
| Downgrade PyTorch | Would fix ABI mismatch | Breaks other packages, not practical |

## The Fix

**Uninstall `mamba-ssm` and its dependency `causal-conv1d`:**

```bash
pip uninstall mamba-ssm causal-conv1d -y
```

This is the correct fix because:

1. `mamba-ssm` is not needed for Granite (a transformer model).
2. Unsloth's internal code already handles the case where `mamba-ssm` is not installed — the `compile_mamba_ssm()` function in `unsloth_zoo/compiler.py` has a bare `except:` that catches `ModuleNotFoundError` and returns `False` gracefully.
3. No other part of the training pipeline depends on `mamba-ssm`.

### Post-fix behavior

After removing `mamba-ssm`, importing `from unsloth import FastLanguageModel` succeeds. You will see warnings about Flash Attention also being incompatible (same PyTorch ABI issue with `flash_attn_2_cuda.so`). Unsloth handles this automatically by falling back to standard PyTorch attention, which the Unsloth team has benchmarked as having negligible performance impact.

## Key Takeaway

The suggested fix in the error message (`pip install 'training-hub[lora]'`) is misleading. The packages are already installed — the problem is that a pre-compiled CUDA extension is binary-incompatible with the current PyTorch version. When troubleshooting "undefined symbol" errors in `.so` files, the relevant question is: **was this extension compiled against the same PyTorch version that is currently installed?** If not, the options are to recompile from source or remove the package if it is not actually needed.
