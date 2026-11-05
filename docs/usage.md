# Running The Interface Code

Run commands from the repository root with `PYTHONPATH` containing `code/`.
All paths below are user-provided assets; these commands are templates, not
executed historical launch records. Keep train/validation/test splits fixed.

## Inputs

An embedding cache is a trusted PyTorch file containing a dictionary with
`cell_ids` (ordered strings) and `embeddings` (an N-by-1280 float tensor).
Use the same preprocessing and normalization convention across splits.
Record JSON files contain `cell_id`, `target_text`, and labels such as
`cell_type`, `mid_cell_type`, and `organ`. The trainer expects a JSON list.
Its `--sft_data` filename must end in `_train_split.json`; the corresponding
`_val_split.json` is resolved by replacing that suffix.

Align IDs and rows explicitly before running. Legacy `source_index` values
are cache-specific, not portable identifiers. The legacy record builder does
not independently prove every supplied index matches its cell ID. Check that
mapping before use; do not treat a successful process exit as this validation.

## CPU Architecture Check

```bash
python -m unittest discover -s tests -v
```

## Training Entry Point

```bash
python code/scripts_v3/train_v3.py \
  --sft_data data/example_train_split.json \
  --uce_train data/train_uce.pt --uce_val data/val_uce.pt \
  --llm_path /path/to/Qwen2.5-14B-Instruct \
  --input_emb_dim 1280 --projector_hidden 1024 \
  --num_cell_tokens 32 --num_mixer_layers 4 --num_heads 8 \
  --llm_dtype bf16 --cell_token_mode resize \
  --output_dir outputs/example_training
```

This selects architecture and precision only. It is **not the paper's full
two-stage recipe**. Set loss coefficients, sampling, schedule, batch size,
selection rule, and Stage-I/II initialization from the intended run's manifest.
Do not assume default losses match a reported run. `--resume_projector` loads
Projector weights; it is not exact optimizer/scheduler/RNG resumption.
Generic extra-task options in the archived trainer do not establish paper
results for trajectory prediction, gene networks, or perturbation prediction.

The training-split language-record builder is available through:

```bash
python code/scripts_v3/build_task1_grounded_language_records.py \
  --records data/example_train_split.json --embedding-cache data/train_uce.pt \
  --out outputs/train_language.json --summary-json outputs/train_language_summary.json \
  --num-cell-tokens 32 --seed 42
```

Use the builder's output as preformatted extra-task records through the
trainer's extra-task options, not as a replacement for its base SFT JSON list.
Its rationale targets are family-consistent templates, not molecular mechanisms.

## Language Controls

```bash
python code/scripts_v3/evaluate_task1_open_questions.py \
  --records data/example_test_split.json --uce_cache data/test_uce.pt \
  --llm_path /path/to/Qwen2.5-14B-Instruct \
  --projector_ckpt checkpoints/projector.pt \
  --input_emb_dim 1280 --projector_hidden 1024 \
  --num_cell_tokens 32 --num_mixer_layers 4 --num_heads 8 \
  --llm_dtype bf16 --cell_token_mode resize \
  --control correct --max_samples 500 --seed 42 \
  --out_prefix outputs/control_correct
```

Repeat with `shuffle`, `zero`, and `absent`, using distinct output prefixes and
identical records, seed, sampling cap, task set, model, and decoding settings.
`zero` passes a zero UCE vector through the Projector (biases remain active);
it does not make the resulting token embeddings zero. `absent` omits the cell
token interface. The legacy `shuffle` draws another row from the sampled pool
with replacement, so self-assignment is possible; it is not a derangement. Check
the emitted sample/control records before aggregation. This command uses the
historical parser and masks, not the revised reassessment protocol.

## Identity Retention

Extract pooled features for each split separately:

```bash
python code/scripts_v3/extract_projector_features_configurable.py \
  --projector-ckpt checkpoints/projector.pt --uce-cache data/test_uce.pt \
  --out-features outputs/test_projector.pt \
  --input-dim 1280 --hidden 1024 --num-cell-tokens 32 --output-dim 5120 \
  --num-mixer-layers 4 --num-heads 8 --dtype fp32 --device cpu
```

For the selected large configuration this is not a tiny-memory CPU operation.
Use `--inspect-only` to inspect architecture without extracting features.
The head trainer takes three record files and aligned feature files; inspect
`python code/scripts_v3/train_fixed_split_label_head.py --help` for exact flags.
Raw UCE and pooled Projector features must use the same splits, labels, head
family, selection budget, and evaluation denominator for a matched comparison.
Training selects on validation; never tune settings using held-out test scores.

Gene-text scripts and aggregators expose `--help`. Their inputs are independent
evidence serializations and output tables, not pretrained author baselines.
