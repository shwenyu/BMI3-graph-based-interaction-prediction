"""Small hand examples and complete experiment checks. Run: python code/check.py."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import data as data_module
from data import Dataset, align_square, candidate_masks, load_dataset, read_matrix, split_edges
from data import project_path, recorded_path
from models import (drug_only, protein_only, neighbour_scores, two_sided,
                    fuse_scores, dti_only, target_degree, random_scores)
from benchmark import (ablation_scores, aggregate, evaluate, main, ranking_metrics,
                       parse_args, save_csv, topk_scores, tune_ablation)
from plot import METRICS, read_results


class DataChecks(unittest.TestCase):
    def test_split_preserves_known_edges(self):
        original = np.ones((4, 5), dtype=np.uint8)
        before = original.copy()
        train, validation, test = split_edges(original, 42)
        self.assertEqual([int(x.sum()) for x in (train, validation, test)], [16, 2, 2])
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(train.astype(int) + validation + test, original)
        self.assertFalse(np.any(train & validation | train & test | validation & test))
        for actual, repeated in zip((train, validation, test), split_edges(original, 42)):
            np.testing.assert_array_equal(actual, repeated)
        for bad in (np.array([[2]]), np.array([]), np.array([1, 0]), np.eye(3)):
            with self.assertRaises(ValueError):
                split_edges(bad, 42)
        for fractions in ((0, .1), (.5, .5), (float("nan"), .1)):
            with self.assertRaises(ValueError):
                split_edges(original, 42, *fractions)

    def test_candidates_exclude_other_positives(self):
        original = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        validation, test = np.zeros((3, 3), bool), np.zeros((3, 3), bool)
        validation[0, 1], test[2, 2] = True, True
        vc, tc = candidate_masks(original, validation, test)
        self.assertEqual((int(vc.sum()), int(tc.sum())), (6, 6))
        self.assertFalse(vc[2, 2] or tc[0, 1])
        np.testing.assert_array_equal(vc, (original == 0) | validation)
        with self.assertRaises(ValueError):
            candidate_masks(original, validation, validation)
        validation[0, 2] = True
        with self.assertRaises(ValueError):
            candidate_masks(original, validation, test)

    def test_alignment_uses_both_identifier_axes(self):
        matrix = np.array([[.25, 1], [1, .25]])
        np.testing.assert_array_equal(align_square(("b", "a"), ("a", "b"), matrix, ("a", "b")),
                                      [[1, .25], [.25, 1]])
        for rows, ids in ((("a", "a"), ("a", "b")), (("b", "a"), ("a", "c"))):
            with self.assertRaises(ValueError):
                align_square(rows, ("a", "b"), matrix, ids)

    def test_reader_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.txt"
            for content in ("", "a b\nr 1\n", "a a\nr 1 2\n", "a b\nr 1 nan\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_matrix(path)

    def test_custom_data_transpose_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "custom_admat_dgc.txt").write_text("d1 d2\np1 1 0\np2 0 1\n")
            (root / "custom_simmat_dc.txt").write_text("d2 d1\nd1 .2 1\nd2 1 .4\n")
            (root / "custom_simmat_dg.txt").write_text("p1 p2\np1 1 .5\np2 .5 1\n")
            data = load_dataset(root, "custom")
            self.assertEqual((data.drugs, data.proteins), (("d1", "d2"), ("p1", "p2")))
            np.testing.assert_array_equal(data.interactions, np.eye(2))
            np.testing.assert_allclose(data.drug_similarity, [[0, .3], [.3, 0]])
            self.assertEqual(len(data.audit["input_sha256"]), 3)
            self.assertTrue(all(len(h) == 64 for h in data.audit["input_sha256"].values()))
            json.dumps(data.audit)
            for prefix in ("../custom", "a/b", ".", ""):
                with self.assertRaises(ValueError):
                    load_dataset(root, prefix)


class ModelChecks(unittest.TestCase):
    def test_baseline_and_active_neighbour_denominators(self):
        graph = np.array([[0], [1], [0]])
        similarity = np.array([[1, .8, .2], [.8, 1, .4], [.2, .4, 1]])
        self.assertAlmostEqual(drug_only(graph, similarity)[0, 0], .8)
        scores, available = neighbour_scores(graph, similarity)
        self.assertEqual(scores[0, 0], 1)
        np.testing.assert_array_equal(available, [True, False, True])
        self.assertEqual(similarity[0, 0], 1)
        np.testing.assert_allclose(protein_only(graph.T, similarity), drug_only(graph, similarity).T)

    def test_power_and_identifier_ties(self):
        graph = np.array([[0, 0], [1, 0], [0, 1]])
        similarity = np.array([[0, .5, .5], [.5, 0, .1], [.5, .1, 0]])
        ids, order = ["query", "z", "a"], [2, 0, 1]
        scores, _ = neighbour_scores(graph, similarity, ids, k=1)
        np.testing.assert_allclose(scores[0], [0, 1])
        permuted, _ = neighbour_scores(graph[order], similarity[np.ix_(order, order)],
                                       [ids[j] for j in order], k=1)
        np.testing.assert_allclose(permuted, scores[order])
        similarity[0, 1], similarity[0, 2] = .8, .2
        scores, _ = neighbour_scores(graph, similarity, power=2)
        np.testing.assert_allclose(scores[0], [16 / 17, 1 / 17])

    def test_loop_and_numpy_scores_agree(self):
        rng = np.random.default_rng(17)
        graph = (rng.random((8, 5)) > .7).astype(int)
        similarity = rng.random((8, 8))
        for k in (None, 1, 3, 20):
            fast, available = neighbour_scores(graph, similarity, k=k, power=2)
            slow, other = neighbour_scores(graph, similarity, k=k, power=2, backend="reference")
            np.testing.assert_allclose(fast, slow, atol=1e-14)
            np.testing.assert_array_equal(available, other)
        np.testing.assert_allclose(neighbour_scores(graph, similarity)[0],
                                   neighbour_scores(graph, similarity, k=100)[0])

    def test_fusion_availability_and_endpoints(self):
        drug, protein = np.full((2, 2), .8), np.full((2, 2), .2)
        np.testing.assert_allclose(two_sided(drug, protein, .75), .65)
        np.testing.assert_allclose(fuse_scores(drug, protein, .5, [True, False], [False, True]),
                                   [[.8, .5], [0, .2]])
        np.testing.assert_allclose(fuse_scores(drug, protein, 1, [True, False], [True, True]),
                                   [[.8, .8], [0, 0]])
        np.testing.assert_allclose(fuse_scores(drug, protein, 0, [True, True], [False, True]),
                                   [[0, .2], [0, .2]])

    def test_three_step_paths_and_controls(self):
        graph = np.array([[1, 0], [1, 1]])
        np.testing.assert_array_equal(dti_only(graph), [[0, 1], [0, 0]])
        np.testing.assert_array_equal(target_degree(graph), [[2, 1], [2, 1]])
        np.testing.assert_array_equal(random_scores(graph.shape, 4), random_scores(graph.shape, 4))

    def test_zero_support_and_invalid_settings(self):
        graph = np.zeros((2, 3))
        scores, available = neighbour_scores(graph, np.ones((2, 2)))
        np.testing.assert_array_equal(scores, graph)
        self.assertFalse(available.any())
        np.testing.assert_array_equal(drug_only(graph, np.zeros((2, 2))), graph)
        for kwargs in ({"k": 0}, {"k": True}, {"power": 0}, {"power": float("inf")},
                       {"ids": ["same", "same"]}, {"backend": "bad"}):
            with self.assertRaises(ValueError):
                neighbour_scores(graph, np.ones((2, 2)), **kwargs)
        for bad in (np.ones((2, 3)), [[0, -1], [1, 0]], [[0, float("nan")], [1, 0]]):
            with self.assertRaises(ValueError):
                drug_only(graph, bad)
        with self.assertRaises(ValueError):
            dti_only([[.5]])
        with self.assertRaises(ValueError):
            two_sided(graph, graph, 1.1)
        with self.assertRaises(ValueError):
            fuse_scores(graph, graph, .5, [True], [True, True, True])


class MetricChecks(unittest.TestCase):
    def test_hand_rankings_and_ties(self):
        result = ranking_metrics([1, 0, 1, 0], [.9, .8, .7, .6])
        self.assertAlmostEqual(result["ap"], 5 / 6)
        self.assertAlmostEqual(result["auroc"], 3 / 4)
        for labels, scores, ap, auc in (([1, 0], [.5, .5], .5, .5), ([0, 1], [2, 1], .5, 0)):
            result = ranking_metrics(labels, scores)
            self.assertEqual((result["ap"], result["auroc"]), (ap, auc))

    def test_undefined_metrics_remain_none(self):
        result = ranking_metrics([0, 0], [2, 1])
        self.assertIsNone(result["ap"])
        self.assertIsNone(result["auroc"])
        result = ranking_metrics([1, 1], [2, 1])
        self.assertEqual(result["ap"], 1)
        self.assertIsNone(result["auroc"])
        pooled, rows = evaluate(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), ("a", "b"))
        self.assertIsNone(pooled["macro_top_precision"])
        self.assertIsNone(pooled["macro_top_recall"])
        self.assertEqual((pooled["eligible_drugs"], len(rows)), (0, 2))

    def test_top_ties_use_ids_not_labels(self):
        scores, hidden = np.zeros((1, 3)), np.array([[1, 0, 0]], bool)
        candidates = np.ones_like(hidden)
        metrics, rows = evaluate(scores, hidden, candidates, ("b", "a", "c"), 1)
        self.assertEqual(rows[0]["hits"], 0)
        order = [2, 0, 1]
        repeated, _ = evaluate(scores[:, order], hidden[:, order], candidates[:, order], ("c", "b", "a"), 1)
        self.assertEqual(metrics, repeated)

    def test_precision_denominator_and_macro_population(self):
        metrics, rows = evaluate([[.8, .2], [.8, .2]], [[1, 0], [0, 0]], [[1, 1], [1, 1]], ("a", "b"), 10)
        self.assertEqual((metrics["macro_top_precision"], metrics["macro_top_recall"]), (.5, 1))
        self.assertEqual((metrics["eligible_drugs"], metrics["drugs_without_hidden"]), (1, 1))
        self.assertIsNone(rows[1]["recall"])

    def test_invalid_metrics_inputs(self):
        for labels, scores in (([2, 0], [1, 0]), ([1], [float("nan")]), ([], []), ([1, 0], [1])):
            with self.assertRaises(ValueError):
                ranking_metrics(labels, scores)
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluate([[1, 0]], [[1, 0]], [[1, 1]], ("a", "b"), n)
        with self.assertRaises(ValueError):
            evaluate([[1, float("inf")]], [[1, 0]], [[1, 0]], ("a", "b"))
        with self.assertRaises(ValueError):
            evaluate([[1, 0]], [[1, 0]], [[0, 1]], ("a", "b"))


class PlotChecks(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def row(self, dataset="nr", method="random"):
        row = dict(dataset=dataset, method=method, runs=5, top_n=10)
        for metric in METRICS:
            row[metric + "_mean"], row[metric + "_sd"] = .25, .1
        return row

    def save(self, rows):
        with (self.root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_blank_metrics_stay_undefined(self):
        row = self.row()
        row["ap_mean"], row["ap_sd"] = "", ""
        self.save([row])
        result = read_results(self.root)["nr"]["random"]
        self.assertIsNone(result["ap_mean"])
        self.assertEqual((result["runs"], result["auroc_sd"]), (5, .1))

    def test_invalid_numeric_values(self):
        for value in ("nan", "inf", "-0.1", "1.01"):
            row = self.row()
            row["ap_mean"] = value
            self.save([row])
            with self.assertRaises(ValueError):
                read_results(self.root)

    def test_missing_columns_duplicates_and_method_mismatch(self):
        row = self.row()
        del row["ap_sd"]
        for rows in ([row], [self.row(), self.row()], [self.row(), self.row("e", "two-sided")]):
            self.save(rows)
            with self.assertRaises(ValueError):
                read_results(self.root)

    def test_cutoff_and_time_validation(self):
        row = self.row("e")
        row["top_n"] = 5
        self.save([self.row(), row])
        with self.assertRaises(ValueError):
            read_results(self.root)
        row = self.row()
        row["score_seconds_mean"] = -1
        self.save([row])
        with self.assertRaises(ValueError):
            read_results(self.root)


class RunChecks(unittest.TestCase):
    def test_complete_run_and_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            drugs, proteins = tuple(f"d{i}" for i in range(5)), tuple(f"p{i}" for i in range(5))
            graph = np.eye(5, dtype=int) + np.roll(np.eye(5, dtype=int), 1, axis=1)
            for suffix, rows, columns, matrix in (
                ("admat_dgc", drugs, proteins, graph),
                ("simmat_dc", drugs, drugs, np.ones((5, 5))),
                ("simmat_dg", proteins, proteins, np.ones((5, 5)))):
                lines = ["\t".join(columns)]
                lines.extend("\t".join([name, *map(str, values)]) for name, values in zip(rows, matrix))
                (root / f"tiny_{suffix}.txt").write_text("\n".join(lines) + "\n")
            output = root / "results"
            args = ["--dataset", "tiny", "--seed", "0", "--input", str(root), "--output", str(output),
                    "--ks", "2", "--powers", "1", "--alphas", ".5", "--timing-repeats", "1",
                    "--compare-backends", "--no-plots"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            tables = {}
            for name in ("metrics", "summary", "parameters", "splits", "predictions", "timing"):
                with (output / f"{name}.csv").open(newline="") as handle:
                    tables[name] = list(csv.DictReader(handle))
            self.assertEqual({name: len(rows) for name, rows in tables.items()},
                             dict(metrics=7, summary=7, parameters=2, splits=10, predictions=15, timing=2))
            self.assertTrue(all(row["selected"] == "True" for row in tables["parameters"]))
            receipt = json.loads((output / "experiment.json").read_text())
            self.assertEqual(len(receipt["runs"]), 1)
            run = receipt["runs"][0]
            self.assertEqual((run["train"], run["validation"], run["test"], run["test_candidates"]), (8, 1, 1, 16))
            self.assertEqual(len(run["data"]["input_sha256"]), 3)
            self.assertEqual(len(receipt["source_sha256"]), 64)
            for row in tables["predictions"]:
                self.assertEqual(graph[drugs.index(row["drug"]), proteins.index(row["protein"])], 0)
                self.assertEqual(row["seed"], "0")
            with self.assertRaises(ValueError):
                main(args)

    def test_sample_sd_and_defined_metric_counts(self):
        common = dict(dataset="tiny", method="random", top_n=10, auroc=None,
                      macro_top_precision=.2, macro_top_recall=.4, score_seconds=.01)
        result = aggregate([dict(common, ap=.2), dict(common, ap=.4)])[0]
        self.assertAlmostEqual(result["ap_mean"], .3)
        self.assertAlmostEqual(result["ap_sd"], .2 / np.sqrt(2))
        self.assertEqual((result["ap_n"], result["auroc_n"]), (2, 0))
        self.assertIsNone(result["auroc_mean"])


class StudyChecks(unittest.TestCase):
    def sample(self):
        graph = np.array([[0, 0], [1, 0], [0, 1]])
        similarity = np.array([[0, .8, .2], [.8, 0, .4], [.2, .4, 0]])
        return Dataset(("d0", "d1", "d2"), ("p0", "p1"), graph,
                       similarity, np.zeros((2, 2)), {})

    def test_incremental_stage_scores_by_hand(self):
        data = self.sample()
        # The protein side has no weights. Plain fusion halves the drug vote;
        # availability fusion uses the supported drug side at full weight.
        cases = (("active", "", 1, [.4, .1]),
                 ("topk", 1, 1, [.5, 0]),
                 ("power", 2, 2, [8 / 17, .5 / 17]),
                 ("availability", 2, 2, [16 / 17, 1 / 17]))
        for stage, k, power, expected in cases:
            setting = dict(stage=stage, k=k, power=power, alpha=.5)
            actual = ablation_scores(data.interactions, data, setting)
            np.testing.assert_allclose(actual[0], expected, atol=1e-14)

    def test_ablation_endpoints_preserve_existing_methods(self):
        data = self.sample()
        setting = dict(stage="baseline", k="", power=1, alpha=.25)
        baseline = two_sided(drug_only(data.interactions, data.drug_similarity),
                             protein_only(data.interactions, data.protein_similarity), .25)
        np.testing.assert_array_equal(ablation_scores(data.interactions, data, setting), baseline)
        setting = dict(stage="availability", k=2, power=2, alpha=.75)
        np.testing.assert_array_equal(ablation_scores(data.interactions, data, setting),
                                      topk_scores(data.interactions, data, setting))

    def test_validation_ties_follow_predeclared_grid(self):
        graph = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
        validation = np.zeros((3, 3), dtype=bool)
        validation[0, 1] = True
        data = Dataset(("d0", "d1", "d2"), ("p0", "p1", "p2"),
                       graph.copy(), np.zeros((3, 3)), np.zeros((3, 3)), {})
        candidates = ~graph
        before = graph.copy()
        selected, trials = tune_ablation(graph, data, validation, candidates,
                                         [.75, .25], [2, 1], [2, 1])
        self.assertEqual(set(selected), {"baseline", "active", "topk", "power", "availability"})
        self.assertTrue(all(row["alpha"] == .75 for row in selected.values()))
        self.assertEqual(selected["topk"]["k"], 2)
        # The study explicitly keeps the preceding power=1 configuration first.
        self.assertEqual(selected["power"]["power"], 1)
        self.assertEqual(selected["availability"]["power"], 1)
        # All scores are zero, so grouped AP is just the positive prevalence.
        expected_ap = validation[candidates].sum() / candidates.sum()
        self.assertTrue(all(abs(row["validation_ap"] - expected_ap) < 1e-14 for row in trials))
        np.testing.assert_array_equal(graph, before)
        np.testing.assert_array_equal(data.interactions, before)

    def test_complete_optional_study_is_separate_and_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = tuple(f"d{i}" for i in range(5)), tuple(f"p{i}" for i in range(5))
            graph = np.eye(5, dtype=int) + np.roll(np.eye(5, dtype=int), 1, axis=1)
            for suffix, rows, columns, matrix in (
                ("admat_dgc", ids[0], ids[1], graph),
                ("simmat_dc", ids[0], ids[0], np.ones((5, 5))),
                ("simmat_dg", ids[1], ids[1], np.ones((5, 5)))):
                lines = ["\t".join(columns)]
                lines.extend("\t".join([name, *map(str, values)]) for name, values in zip(rows, matrix))
                (root / f"nr_{suffix}.txt").write_text("\n".join(lines) + "\n")
            output = root / "validation"
            args = ["--study", "--dataset", "nr", "--seed", "0", "--stability-seeds", "2",
                    "--input", str(root), "--output", str(output), "--ks", "2", "--powers", "1",
                    "--alphas", ".5", "--timing-repeats", "1", "--no-plots"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            tables = {}
            for name in ("seed42", "stability", "ablation", "ablation_parameters", "paired"):
                with (output / f"{name}.csv").open(newline="") as handle:
                    tables[name] = list(csv.DictReader(handle))
            self.assertEqual({name: len(rows) for name, rows in tables.items()},
                             dict(seed42=5, stability=8, ablation=5, ablation_parameters=5, paired=16))
            self.assertEqual({row["seed"] for row in tables["seed42"]}, {"42"})
            self.assertEqual({row["seed"] for row in tables["stability"]}, {"0", "1"})
            self.assertEqual({row["stage"] for row in tables["ablation"]},
                             {"baseline", "active", "topk", "power", "availability"})
            for name in ("seed42", "stability", "ablation"):
                self.assertTrue(all(int(row["positives"]) == 1 and int(row["background"]) == 15
                                    for row in tables[name]))
            self.assertFalse((output / "metrics.csv").exists())
            by_stage = {row["stage"]: row for row in tables["ablation"]}
            for row in tables["paired"]:
                delta = float(by_stage[row["stage"]][row["metric"]]) - float(by_stage[row["previous"]][row["metric"]])
                self.assertAlmostEqual(float(row["mean_delta"]), delta)
                self.assertEqual(int(row["runs"]), 1)
                self.assertEqual(int(row["wins"]) + int(row["ties"]) + int(row["losses"]), 1)
            receipt = json.loads((output / "study.json").read_text())
            self.assertEqual(len(receipt["source_sha256"]), 64)
            self.assertEqual(receipt["config"]["stability_seeds"], 2)
            with self.assertRaises(ValueError):
                main(args)


class PathChecks(unittest.TestCase):
    def write_tiny_data(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        drugs, proteins = tuple(f"d{i}" for i in range(5)), tuple(f"p{i}" for i in range(5))
        graph = np.eye(5, dtype=int) + np.roll(np.eye(5, dtype=int), 1, axis=1)
        for suffix, rows, columns, matrix in (
            ("admat_dgc", drugs, proteins, graph),
            ("simmat_dc", drugs, drugs, np.ones((5, 5))),
            ("simmat_dg", proteins, proteins, np.ones((5, 5)))):
            lines = ["\t".join(columns)]
            lines.extend("\t".join([name, *map(str, values)]) for name, values in zip(rows, matrix))
            (folder / f"tiny_{suffix}.txt").write_text("\n".join(lines) + "\n")
        return graph

    def test_cli_paths_follow_project_not_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "project"
            unrelated = root.parent / "unrelated"
            root.mkdir()
            unrelated.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(unrelated)
                with mock.patch.object(data_module, "ROOT", root):
                    normal = parse_args([])
                    study = parse_args(["--study"])
                    custom = parse_args(["--input", "input", "--output", "output/new run"])
                    outside = root.parent / "external data"
                    absolute = parse_args(["--input", str(outside)])
                    self.assertEqual(normal.input, root / "input")
                    self.assertEqual(normal.output, root / "output")
                    self.assertEqual(study.output, root / "output/validation")
                    self.assertEqual(custom.output, root / "output/new run")
                    self.assertEqual(absolute.input, outside)
                    self.assertEqual(project_path("input"), root / "input")
            finally:
                os.chdir(previous)

    def test_recorded_paths_are_project_relative(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "project"
            with mock.patch.object(data_module, "ROOT", root):
                self.assertEqual(recorded_path(root / "input"), "input")
                self.assertEqual(recorded_path("output/validation"), "output/validation")
                self.assertEqual(recorded_path(root.parent / "shared"), "../shared")
                self.assertFalse(Path(recorded_path(root / "input")).is_absolute())

    def test_relative_file_operations_use_project_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "project"
            expected = self.write_tiny_data(root / "input")
            (root / "output").mkdir()
            unrelated = root.parent / "unrelated"
            unrelated.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(unrelated)
                with mock.patch.object(data_module, "ROOT", root):
                    rows, columns, matrix = read_matrix("input/tiny_admat_dgc.txt")
                    self.assertEqual((rows[0], columns[0]), ("d0", "p0"))
                    np.testing.assert_array_equal(matrix, expected)
                    np.testing.assert_array_equal(load_dataset("input", "tiny").interactions, expected)
                    save_csv("output/example.csv", [dict(value=5)])
                    summary = dict(dataset="tiny", method="random", runs=1, top_n=10)
                    for metric in METRICS:
                        summary[metric + "_mean"], summary[metric + "_sd"] = .25, ""
                    save_csv("output/summary.csv", [summary])
                    self.assertEqual(read_results("output")["tiny"]["random"]["ap_mean"], .25)
                self.assertTrue((root / "output/example.csv").is_file())
                self.assertFalse((unrelated / "output").exists())
            finally:
                os.chdir(previous)

    def test_copied_project_runs_from_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "copied project"
            code = root / "code"
            code.mkdir(parents=True)
            for source in Path(__file__).resolve().parent.glob("*.py"):
                shutil.copy2(source, code / source.name)
            self.write_tiny_data(root / "input")
            unrelated = root.parent / "unrelated working directory"
            unrelated.mkdir()
            command = [sys.executable, str(code / "benchmark.py"), "--dataset", "tiny", "--seed", "0",
                       "--ks", "2", "--powers", "1", "--alphas", ".5", "--timing-repeats", "1", "--no-plots"]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            for extra in ([], ["--study", "--stability-seeds", "2"]):
                completed = subprocess.run(command + extra, cwd=unrelated, env=environment,
                                           capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            normal = json.loads((root / "output/experiment.json").read_text())
            study = json.loads((root / "output/validation/study.json").read_text())
            self.assertEqual((normal["config"]["input"], normal["config"]["output"]), ("input", "output"))
            self.assertEqual((study["config"]["input"], study["config"]["output"]), ("input", "output/validation"))
            self.assertEqual(normal["runs"][0]["test_candidates"], 16)
            self.assertEqual({r["candidates"] for r in study["runs"]}, {16})
            with (root / "output/metrics.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 7)
            self.assertTrue(all(np.isfinite(float(r["ap"])) for r in metrics))
            self.assertTrue((root / "output/predictions.csv").is_file())
            self.assertFalse((unrelated / "output").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
