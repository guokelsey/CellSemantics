# CellSemantics

Source code for adapting single-cell representations to a frozen language
decoder while assessing identity retention and cell-dependent language readout.
The interface maps a precomputed UCE vector to continuous cell tokens consumed
by Qwen2.5-14B-Instruct. UCE and decoder parameters are frozen; language-loss
gradients still propagate through the decoder into the trainable Projector.

## Release Scope

This is a source-only release of the archived interface implementation, its
training/evaluation entry points, restored local helper dependencies, and CPU
packaging tests. It is **not a self-contained numerical replication package**:
raw data, split records, embedding caches, trained weights, generated answers,
and the paper PDF are not included. Tests do not reproduce the paper's scores.
Existing names such as `SAGA`, `v3`, and `Task1` are retained for compatibility.

The selected interface uses UCE dimension 1280, Projector hidden width 1024,
32 cell tokens, four Transformer mixer layers, eight attention heads, and
decoder embedding width 5120. Several constructors default to smaller models;
pass the selected dimensions explicitly. There is no cross-attention layer
in this Projector. The optional `saga_xattn_runtime.py` is archived support
code, not the evaluated paper architecture.

## Installation

Use Python 3.10 or 3.11 for a new environment. Install a PyTorch build suitable
for your hardware before installing the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

`requirements.txt` specifies a compatibility target, not an authenticated
historical environment. CPU tests require no model downloads. Actual language
training/inference requires separately obtained Qwen assets and suitable GPU
memory. UCE embeddings must already be computed using the upstream UCE model.
Consult [usage](docs/usage.md) for input schemas and commands.

## Check The Installation

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
python code/scripts_v3/train_v3.py --help
python code/scripts_v3/evaluate_task1_open_questions.py --help
```

The tests use tiny synthetic tensors and a toy frozen linear decoder, not Qwen
or biological data. CLI help checks validate imports and arguments, not GPU
execution. In the initial release, Transformers-dependent CLI checks remain
unverified because the dependency download timed out; see
[validation scope](docs/validation.md) and the per-entry-point receipt.

## Entry Points

| Purpose | File under `code/` |
| --- | --- |
| Continuous token Projector | `models_v3/uce_projector.py` |
| Frozen decoder wrapper | `models_v3/sagav3_llm.py` |
| Projector training and loss components | `scripts_v3/train_v3.py` |
| Training-split language record builder | `scripts_v3/build_task1_grounded_language_records.py` |
| Correct/shuffle/zero/absent evaluation | `scripts_v3/evaluate_task1_open_questions.py` |
| Multi-run control aggregation | `scripts_v3/collect_task1_null_controls_multiseed.py` |
| Projector feature extraction | `scripts_v3/extract_projector_features_configurable.py` |
| Matched linear/MLP label heads | `scripts_v3/train_fixed_split_label_head.py` |
| Gene-text evidence builder/evaluation | `scripts_v3/build_task1_gene_text_evidence.py`, `scripts_v3/evaluate_task1_gene_text_questions.py` |
| Gene-text/interface aggregation | `scripts_v3/collect_task1_raw_gene_interface_multiseed.py` |

## Interpretation And Reuse

Read [protocol scope](docs/protocol_scope.md) before comparing outputs. In
particular, the archived scorer retains historical lexical family matching;
it is not the revised label/mask and strict-parser reassessment protocol.
Changing those rules creates a new evaluation, not a corrected copy of old
scores. Five reported Stage-II continuations share a Stage-I initialization;
they are not five independent end-to-end training histories.

Only load checkpoints and embedding caches from trusted sources. Some archived
entry points use `torch.load` and are not a secure loader for untrusted files.
Do not commit private cell records, credentials, or model assets to this repo.

Code origins and packaging-only changes are described in
[code provenance](docs/code_provenance.md). No new software license is declared
in this release; public visibility does not itself grant a reuse license.
Upstream model, software, and dataset terms apply separately.
