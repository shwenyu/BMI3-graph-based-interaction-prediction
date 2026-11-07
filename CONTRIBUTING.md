# Development and authorship

This scaffold was AI-assisted on 2026-11-07 at the group's request. It supplies
data plumbing, benchmark/evaluation infrastructure, documentation and baseline checks.
The central similarity-based scoring methods are intentionally unimplemented.
Students must understand, author, verify and explain their project algorithm and
follow the course's current AI-use/disclosure rules. Do not describe this scaffold
or its baseline outputs as a completed student algorithm or final scientific result.

Use a feature branch, keep the original paper-work templates, and avoid committing
raw downloads, private course feedback, generated experiment directories or credentials.
Record changes to the scoring rule and experimental protocol before comparing results.
Before changing main, review the branch diff and run the contract tests.

Small tests should establish independent hand-calculated facts: masking invariants,
ID permutation handling, ranking ties, cold-start behaviour and same-rule optimization
equivalence. Do not select parameters or cases using final hidden labels.

An optional GitHub Actions example is provided in `docs/ci-checks.yml.example`.
Copy it to `.github/workflows/checks.yml` only using a connection authorized to manage workflows.
The current branch relies on verified local checks.
