# Infrastructure verification — 2026-11-07

This is a scaffold verification receipt, not a scientific performance claim.

- Python: 3.12.14; NumPy: 2.3.5.
- Downloaded 16 source TXT files; all SHA-256 hashes verified against the recorded manifest.
- All four adjacency matrices align with both similarities and match their independent source edge lists after the documented hsa-colon normalization.
- Six unit tests passed, covering node/edge masking invariants and repeatability, entity holdouts, ID permutations, duplicate IDs, hand-calculated AP/AUROC and tied scores, top-N handling, and visible-only degree scoring.
- Full default baseline experiment completed: four datasets × three random hidden fractions × five seeds × two baselines = 120 evaluations.
- Additional nuclear-receptor smoke experiment completed: three hiding modes × one fraction × one seed × two baselines = six evaluations.
- Every baseline result was checked for candidate/positive/background counts and finite AP/AUROC in [0,1].
- Local complete outputs: `results/infrastructure-baseline-2026-11-07/` and `results/entity-smoke-2026-11-07/` (ignored by Git).
- Baseline runs were made in the scaffold working tree; their environment files accurately mark it dirty and record the starting Git commit. They are infrastructure checks, not frozen final-test results.
- Similarity-based student scoring remains unimplemented. No RWR or scientific accuracy improvement has been claimed.
- GitHub Actions configuration is supplied as `docs/ci-checks.yml.example`, not installed: the GitHub OAuth connection rejected workflow creation because it lacks workflow scope. A repository maintainer may later copy it to `.github/workflows/checks.yml` through an authorized connection. No hosted CI execution is claimed.
