# Slurm templates

These scripts are deliberately generic. Replace partition, account, environment,
resource requests, and experiment configuration for the target cluster.

Recommended dependency chain:

```text
validate -> predict -> explain array -> evaluate -> report
```

Do not submit a large method matrix until one single-example smoke test and one
small-batch run have completed successfully.
