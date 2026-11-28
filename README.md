# BMI3 graph-based interaction prediction

This program ranks unrecorded drug–protein pairs using simple graph and
similarity-based methods. It includes a benchmark that hides recorded edges,
selects parameters using a separate validation split, and evaluates on held-out
test edges. Original zeros are **unlabeled background**, not proven negatives.

This is an AI-assisted implementation. Read [AI_PROVENANCE.md](AI_PROVENANCE.md)
before reusing it. It is not represented as independently student-authored code
or as compliant with the course's submission restrictions.

## Setup and quick run

Use Python 3.10 or newer. From this repository directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python source/run.py --dataset nr --seed 42 --output output/nr-example --compare-backends
python source/plot_benchmark.py --results output/nr-example
```

Use a new output directory for each experiment. The program refuses to overwrite
an existing nonempty benchmark. `python source/run.py --help` lists all options.

## Full benchmark

```sh
python source/run.py --output output/full
python source/plot_benchmark.py --results output/full
```

Defaults are all four supplied datasets (`e`, `ic`, `gpcr`, `nr`), seeds 0–4,
approximately 80/10/10 train/validation/test positives, top ten per-drug ranking,
alpha values 0.5/0.25/0.75, k values 5/10/20/50 and powers 1/2. Each held-out count
is rounded down, leaving the remaining edges in training. Small custom datasets
must have enough positives to leave every partition nonempty.

Seven methods share the same edge split and candidate sets:

| Method | Evidence |
| --- | --- |
| Random | Seeded uniform ranking scores |
| Target degree | Visible interactions per protein |
| DTI-only | Non-backtracking drug→protein→drug→protein paths |
| Drug-only | All non-self chemical-similarity neighbours |
| Protein-only | All non-self protein-similarity neighbours |
| Two-sided | Validation-selected fixed fusion of the two baselines |
| Two-sided Top-k | Active neighbours, Top-k, power weights and availability fusion |

Top-k refers to neighbours, while top ten refers to returned target predictions.
RWR remains an optional research extension and is not included in this program.
No advanced machine-learning framework or imported DTI predictor is used.

## Inputs and outputs

The included inputs are inherited from the `Hanyu-Shou` branch. A custom dataset
can use `--input /path/to/matrices --dataset custom`, with these labelled files:

* `custom_admat_dgc.txt`: binary interaction matrix, either orientation.
* `custom_simmat_dc.txt`: square drug similarity matrix.
* `custom_simmat_dg.txt`: square protein similarity matrix.
* Optional `bind_orfhsa_drug_custom.txt`: protein/drug edge list cross-check.

Every matrix begins with whitespace-separated column IDs, followed by a row ID
and numeric entries on each row. IDs must be unique. Alignment uses identifiers,
not assumed file positions. Similarities must be finite and nonnegative; they
are averaged with their transpose and their diagonals are set to zero. Source
anomalies, transformations and SHA-256 hashes are recorded. Protein IDs in the
optional edge list accept the inherited `hsa:`→`hsa` normalisation.

For custom datasets, use distinct drug/protein ID namespaces (for example,
`drug_1` and `protein_1`). If the two complete ID sets are identical, the loader
chooses drug rows first and cannot infer a transposed orientation from IDs alone.

Each `dataset_seedN/` directory contains `edges.csv`, `validation.csv`, `test.csv`,
`per_drug.csv` and `run.json`. The receipt includes counts, selected settings,
data/code hashes, environment, Git state and timings. `--compare-backends` also
checks loop/vectorised score equivalence and writes `backend_timings.csv`.
Root files `all_runs.csv` and `summary.csv` contain per-run results and means with
sample standard deviations across seeds. Undefined metrics are blank/null.
Figures are PNG/PDF files under `figures/`.

## Rank new candidates after benchmarking

```sh
python source/predict.py --dataset nr \
  --settings output/nr-example/nr_seed42/run.json \
  --output output/nr-predictions --top-n 10
```

This refits the chosen Top-k rule on **all recorded edges**, checks that the input
file hashes match the benchmark, and exports ranked original-zero pairs. Scores
are not calibrated probabilities. These predictions cannot be used to evaluate
the same benchmark holdouts, because those edges have now been restored.

See [docs/METHODS.md](docs/METHODS.md) for formulas, evaluation conventions,
limitations, complexity and timing definitions, and [docs/REFERENCES.md](docs/REFERENCES.md)
for source and course-topic attribution. Assessment templates remain unchanged.
