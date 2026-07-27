# General Explainability Pipeline Template

A reusable, model-agnostic repository scaffold for explainability experiments.
It is intentionally implementation-light: the intern receives a clear
architecture, contracts, configuration system, output schemas, tests, and a
staged implementation plan without being locked to one dataset, model, language,
or explanation method.

The structure prioritises NLP and multilingual work, while the contracts can
also support tabular, vision, speech, and multimodal projects.

## Architecture

```mermaid
flowchart LR
    A[Raw data] --> B[Validate and normalise]
    B --> C[Load or train model]
    C --> D[Run predictions]
    D --> E[Freeze explanation cohort]
    E --> F[Generate explanations]
    F --> G[Normalise and align units]
    G --> H[Evaluate explanations]
    H --> I[Paired statistical comparison]
    I --> J[Error analysis]
    J --> K[Tables, figures and report]
```

## Separation of responsibilities

1. **Dataset adapters** return stable examples and reference evidence.
2. **Model adapters** expose predictions, target scores, tokenisation, attention,
   gradients, hidden states, or other supported capabilities.
3. **Explainers** generate raw method-specific evidence.
4. **Aligners** convert subwords, tokens, spans, features, regions, or examples
   to a shared comparison unit.
5. **Evaluators** measure faithfulness, plausibility, stability, sanity,
   class specificity, agreement, and efficiency.
6. **Reporters** create reproducible tables, figures, manifests, and error cases.

## Repository map

```text
configs/                 YAML experiment configuration
data/                    Local data layout; data are not committed
docs/                    Contracts, protocol, method cards and intern guide
experiments/             Pre-registered experiment cards
notebooks/               Thin inspection notebooks only
scripts/                 Command-line entry points
src/xai_pipeline/        Reusable package code
outputs/                 Generated artefacts organised by run
 tests/                  Unit and dry-run tests
```

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,nlp,notebooks]"
make validate
make test
make dry-run
```

The dry run validates the configuration and writes a frozen run manifest. The
actual data, model, explainer, alignment, evaluation, and reporting components
remain explicit placeholders until assigned.

## Recommended first implementation

1. Implement one dataset adapter.
2. Implement one model adapter and reproduce the original predictions.
3. Freeze a balanced explanation cohort.
4. Implement one cheap diagnostic method, such as raw attention.
5. Implement one gradient method, such as gradient × input.
6. Implement one perturbation method, such as leave-one-out.
7. Align all scores to words or another shared unit.
8. Evaluate faithfulness before plausibility.
9. Add paired confidence intervals, tests, and error analysis.

## Safeguards

- Attention is a diagnostic baseline, not automatically a complete explanation.
- Raw predictions and raw explanation scores are stored separately.
- All methods must explain the same examples, checkpoint, target function, and
  unit of comparison.
- Failed or skipped examples remain visible.
- Human overlap is plausibility evidence, not direct proof of faithfulness.
- Cross-method agreement is not proof that the methods are correct.
- The held-out test set is reserved for final reporting.

See `docs/INTERN_GUIDE.md` and `docs/PIPELINE.md`.
