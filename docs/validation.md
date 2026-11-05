# Validation Scope

The release contains synthetic CPU tests for syntax, restored import symbols,
the shared cell-placeholder literal, prompt slot count, legacy label helper
behavior, tiny Projector tensor shapes, and gradient-path separation.
The frozen-decoder test uses a toy linear layer, not Qwen.

No pretrained weights or biological records are loaded by these tests. A
passing result does not authenticate historical configurations, reproduce
scores, exercise distributed training, or establish GPU/model compatibility.
CLI `--help` checks, when successful, validate real imports and argument
parsing only. A release validation receipt records the checks actually run.

For this initial release, all nine synthetic CPU tests passed on Python 3.9.6
and PyTorch 2.8.0. Transformers/Accelerate installation was not completed
because the dependency download timed out. CLI entry points requiring
Transformers therefore remain unverified locally; they are not counted as
successful import checks. No model download, GPU inference or training ran.
See [the receipt](release_validation.json) for each entry point's status.
