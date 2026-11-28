# Sources and course-topic mapping

## Dataset and prior prototype

* Yamanishi Y, Araki M, Gutteridge A, Honda W, Kanehisa M (2008).
  “Prediction of drug–target interaction networks from the integration of chemical
  and genomic spaces.” Bioinformatics 24(13), i232–i240.
  https://doi.org/10.1093/bioinformatics/btn162
* Original supplement: http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/
* Upstream repository and inherited prototype:
  https://github.com/shwenyu/BMI3-graph-based-interaction-prediction/tree/Hanyu-Shou
  Base commit: `7c0fed98786352e498161f1ec429dd3251936958`.

The inherited input files are used as fixed benchmark data. This project uses
its own random edge train/validation/test protocol; it is not a reproduction of
the paper's model or validation results. Raw similarities are symmetrised and
diagonals removed as documented. Predicted-pair reference files are excluded
from every loader, scoring method, tuning step and evaluation label set.

## Course concepts

The supplied BMI3 materials cover NumPy/matrix operations (Week 1), basic data
structures including heaps (Week 2), exhaustive search (Week 2.2), randomised
algorithms (Week 3), graph traversal/pseudocode (Week 4), greedy approaches
(Weeks 5.2–6), graph representations and algorithms (Week 7), and dynamic
programming (Week 8). This implementation uses arrays, sorting, random splits,
explicit loops and graph paths. It does not import a ready-made DTI predictor.

Weighted voting, availability correction, AP/AUROC evaluation and this exact
fusion design are project choices/extensions, not claimed to be copied course
algorithms. Basic building blocks appearing in lectures do not mean that the
whole implementation is approved for assessed submission. No novelty claim is
made. There is no RWR implementation in this version.

## Libraries and implementation provenance

* Python standard library (argparse, csv, hashlib, json, pathlib, time, unittest):
  https://docs.python.org/3/library/
* NumPy array operations: https://numpy.org/doc/stable/
* Matplotlib plots: https://matplotlib.org/stable/

These are dependency references, not claims that code was copied from their
documentation. New implementation code was AI-assisted as disclosed in
`AI_PROVENANCE.md`; inherited code is identified above. No copied external
algorithm package is used. Before redistribution, check the upstream data and
repository terms; this project does not assert a new license over them.
