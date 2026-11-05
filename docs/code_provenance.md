# Code Provenance

The model, trainer and evaluation entry points were recovered from the
author-supplied archived conference code bundle. They retain their original
module names and numerical operations. Two local dependencies were restored
from compatible author code: `models/prompt_config.py` and four pure label
helpers in `scripts/step1_train_projector.py`. This dependency recovery is not
an authentication of the exact prompt-helper version used by every old run.

Packaging changes:

- Added package initializers and the shared literal `CELL_PLACEHOLDER`.
- Replaced the prompt module's unrelated older model-wrapper import with that
  literal; added postponed annotations for Python 3.9 compatibility.
- Extracted only `parse_rich_label`, `parse_target`, `normalize`, and
  `fuzzy_match` without changing their function bodies; the unrelated older
  training program is not included.
- Replaced one machine-specific path in a training usage example.
- Corrected parameter-count/auxiliary-gradient and legacy-scoring comments;
  added installation, usage, evidence-boundary documentation and tests.

No new scoring parser, training loss, biological result or license was created.
The optional cross-attention runtime remains separate from the paper model.
The release manifest records SHA-256 and Git blob hashes of the public files.

Third-party components include PyTorch, Hugging Face Transformers/Accelerate,
NumPy, pandas and scikit-learn. UCE and Qwen assets must be obtained under
their upstream terms. This repository does not redistribute those assets.
