# Methods and evaluation conventions

## Visible graph and baseline scores

Let A be the visible binary drug-by-protein matrix. S_D and S_P are aligned,
symmetrised similarities with zero diagonals. Drug-only voting uses

`D(i,j) = sum_l S_D(i,l) A(l,j) / sum_l S_D(i,l)`.

Protein-only applies the same rule to A transposed. Zero denominators produce
zero. These baselines include neighbours with zero visible degree in the
denominator, preserving the inherited baseline definition. Fusion is
`alpha * D + (1-alpha) * P`. Alpha is chosen on validation AP, never test AP.

DTI-only counts length-three bipartite paths without immediate backtracking:

`A A^T A - A * (drug_degree + protein_degree - 1)`.

The multiplication in the correction is elementwise. On unseen pairs A(i,j)=0,
so this agrees with the raw three-step path count. Target degree and seeded
uniform scores are simple controls.

## Enhanced Top-k voting

For each query, exclude itself, exclude neighbours with zero visible degree,
and exclude nonpositive weights. Sort by descending similarity, breaking exact
ties by identifier. Retain at most k neighbours, raise weights to a positive
power, and normalise to sum to one. Scaling by the maximum selected weight
before exponentiation prevents overflow without changing normalised scores.

Compute a drug-side and protein-side vote. Let a_D and a_P indicate that the
corresponding query has at least one eligible neighbour. The fused score is

`(alpha*a_D*D + (1-alpha)*a_P*P) / (alpha*a_D + (1-alpha)*a_P)`.

If that denominator is zero, return zero. At alpha=0 or 1, a side explicitly
assigned zero weight is not used as a fallback. k, power and alpha are chosen
jointly on validation AP. Exact trial ties use the first grid entry: defaults
prefer alpha 0.5, k=5 and power=1. These enhanced changes are an ablation bundle;
an improvement cannot be attributed to Top-k alone without separate ablations.

## Leakage prevention

Only original positive edges are partitioned. Validation candidates consist of
validation positives plus original-zero background, excluding test positives.
After settings are frozen, restore validation edges, keeping test positives
hidden. Final test candidates consist of test positives plus original-zero
background. Every method sees identical graphs/candidates for the same run.
Reference predictions supplied with the original paper are never loaded.

The splitter is random **edge hiding**, not a drug/protein entity holdout study.
Receipts count cold drugs/proteins and held-out edges touching them, but these
counts alone do not establish cold-start performance. Supplied similarity
matrices are treated as fixed side information; their upstream construction
must be checked independently if other data or learned similarities are used.

## Metrics

AP and AUROC are pooled over candidate pairs. AP is the noninterpolated sum of
recall gains times precision at the end of each tied score group. AUROC is the
trapezoidal ROC area over those same groups. With no positives AP is undefined;
with either class absent AUROC is undefined. Ties are not broken by labels.

Per-drug rankings use score descending and protein ID ascending. Precision is
hits divided by `min(top_n, candidate_count)`. Recall is hits divided by held-out
positives. Macro averages include only drugs with at least one held-out positive,
and the eligible/excluded counts are exported. Thus these are conditional macro
metrics, not averages over all drugs. With fewer than ten candidates, the output
is not literally fixed-denominator Precision@10. Undefined values remain null.

Across-seed summaries show mean and sample SD (ddof=1); a one-run SD is undefined.
Overlapping random splits are not independent studies or clinical evidence.
Unlabeled pairs may contain true interactions, so metrics measure recorded-edge
recovery and cannot establish biological negative classification accuracy.

## Runtime and complexity

Each scoring timing is the median of `--timing-repeats` calls. Drug-only and
protein-only are measured separately; unrestricted fusion is timed end to end,
including recomputation of both single-sided scores. Enhanced timing includes both neighbour
selection/voting steps and fusion. Loading, tuning and evaluation are excluded
from scoring time; tuning and evaluation have separate measurements. Timings
include allocation/validation and have no explicit warmup. Record thread settings
and hardware before comparisons; no universal speedup is claimed.

With D drugs and P proteins, dense similarities occupy O(D²+P²), and the graph
and score arrays O(DP). Dense single-sided votes cost O(D²P+DP²). Top-k scoring
still scans and sorts each similarity row: O(D² log D + P² log P) selection and
O(DPk) voting, in addition to those dense input arrays. It does not make memory
sparse or guarantee lower runtime. The reference backend uses explicit nested
loops; the vectorised backend computes each selected weighted vote with NumPy.
`--compare-backends` checks identical scores to 1e-12 tolerance before exporting
their timings. DTI-only uses dense matrix products, unsuitable for very large
networks. Start with small inputs when adapting this implementation.
