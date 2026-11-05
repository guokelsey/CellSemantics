# Protocol And Evidence Boundaries

## What The Code Implements

The Projector expands a UCE vector through an MLP, adds slot positions,
mixes the slots with a pre-normalized Transformer encoder, and linearly maps
them to the decoder embedding dimension. Qwen-facing tokens and optional
pooled-token heads share this path. Raw-UCE auxiliary classifiers are a
separate branch and cannot directly train token expansion. Freezing Qwen
parameters does not detach the language loss from Projector inputs.

The selected 1280/1024/32/5120, four-layer, eight-head configuration has
132,114,024 parameters with the default 44/8 raw and pooled auxiliary heads;
125,438,976 belong to the token path. Changing head vocabularies changes the
total. Constructor defaults and optional hierarchy heads are distinct choices.

## Historical Versus Revised Evaluation

The archived open-question evaluator uses substring/keyword family recovery
and lexical label overlap. Those rules can count a mention without proving a
correct canonical field or causal reasoning. The family mapper and endpoint
eligibility should be inspected before reuse. This release does not silently
replace those rules with a stricter parser or rename old outputs as revised
results. Evaluation rules, masks, ID mappings, precision, and model identity
must be versioned together for a new reassessment.

Correct/shuffle/zero/absent comparisons address dependence on assigned cell
inputs under the specified prompt and sample. They do not alone establish
gene-level explanations, out-of-source generalization, unseen-ontology
annotation, or superior performance to complete external interfaces.
Family-consistent rationale templates are not mechanistic ground truth.

## Training And Selection

Five Stage-II continuations share a Stage-I initialization. Variation across
them must not be described as five independent end-to-end training histories.
The generic trainer exposes several possible objectives. Its defaults and
available options do not authenticate which coefficients or checkpoint
selection procedure generated a historical result. Retain run-specific
manifests and validation/selection records when performing new runs.

Relational, contrastive, variance and covariance components are standard
objective families; the code does not imply these are new primitives.
For a two-example contrastive pool with only one other candidate per anchor,
the ordinary denominator contains no competing negative when that candidate
is positive; do not infer a useful contrastive-learning effect merely from
a nonzero configured coefficient.

## Remaining Replication Requirements

Exact score replication also needs approved split definitions, ID-aligned
embedding caches, trained checkpoint hashes, full launch configurations,
selection provenance, generated answers and scoring receipts. These artifacts
are not distributed in this source-only release. Local CPU packaging checks
are not a substitute for GPU inference or those records.
