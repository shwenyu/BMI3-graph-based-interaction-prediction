# BMI3 implementation

The complete run, plotting and prediction instructions are now in the root
`README.md`. Method definitions and timing conventions are in `docs/METHODS.md`.
Authorship is disclosed in `AI_PROVENANCE.md`. The earlier prototype notes below
describe the inherited starting point; the current CLI defaults and seven-method
benchmark supersede them.

## Earlier prototype notes

Run from the branch root with Python 3.10+ and NumPy installed.

    python source/run.py --dataset nr --seed 42

The four datasets are included in `input/`. Use one prefix per run: `e` (enzyme),
`ic` (ion channel), `gpcr` (GPCR), or `nr` (nuclear receptor). Each prefix selects
its own interaction matrix, chemical similarity matrix and protein similarity
matrix. The accompanying edge list is used to check the interaction matrix.
Use the same 8:1:1 splitting procedure for each dataset and report results by class.

The files come from the [Yamanishi et al. (2008) supplement](http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/).
[Section 4.2 and Table 2 of the paper](https://doi.org/10.1093/bioinformatics/btn162) report results by target class using ten-fold cross-validation.
This prototype uses a separate validation split to select the fusion weight.
Files under `input/reference_predictions/` contain the paper's predicted pairs for reference.
Benchmark labels come from the gold-standard interaction matrix.

Output paths include dataset and seed. Use a fresh --output directory to replay a run.
edges.csv records the identity and split of every known interaction.
validation.csv compares three fusion weights on a shared visible graph.
test.csv compares five methods after restoring validation edges.
per_drug.csv lists top-ten hits, precision and recall. Macro averages include drugs
with held-out positives. Original unrecorded pairs are evaluation background.
This AI-assisted prototype uses NumPy broadcasting (W1), edge lists, adjacency
matrices and visible degree (W7). Review the rules and hand-check examples as a group.
