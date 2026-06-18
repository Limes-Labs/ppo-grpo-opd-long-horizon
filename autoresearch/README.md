# AutoResearch Hooks

This directory contains Limes AutoResearch specs and runner configs for the
closed-loop credit-assignment workstream. The configs assume they are launched
from this repository root while pointing `PYTHONPATH` at the sibling
`limes-autoresearch` checkout:

```bash
PYTHONPATH=/Users/francescogiannicola/Documents/LimesLabs/workstreams/limes-autoresearch/repo \
  python3 -m autoresearch_limes validate-spec autoresearch/coverage_gated_credit_spec.json

PYTHONPATH=/Users/francescogiannicola/Documents/LimesLabs/workstreams/limes-autoresearch/repo \
  python3 -m autoresearch_limes run autoresearch/closed_loop_credit_training_config.json \
  --ledger autoresearch/runs/closed_loop_ledger.jsonl

PYTHONPATH=/Users/francescogiannicola/Documents/LimesLabs/workstreams/limes-autoresearch/repo \
  python3 -m autoresearch_limes report-card autoresearch/runs/closed_loop_ledger.jsonl \
  --spec autoresearch/coverage_gated_credit_spec.json \
  --out autoresearch/reports/closed_loop_coverage_gated_card.md
```

The spec is deliberately modest: coverage-gated credit only passes promotion if
it beats the group-total baseline in a low-coverage closed-loop setting and the
run is replayed.
