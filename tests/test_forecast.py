"""Tests for the forecast entry point and the backtest's scoring.

Run from the repository root:  python3 -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest  # noqa: E402
import main as entry  # noqa: E402

RULES_EXAMPLE_ROWS = [[0.27, 0.29, 0.30, 0.31, 0.33], [0.27, 0.29, 0.30, 0.31, 0.33]]


def make_case(markets: dict[str, list[float]], case_id: str = "t00", horizon: int = 240) -> dict:
    return {
        "schema": "makefun/polymarket-case/v1",
        "competition": "polymarket-nvidia-ceo-forecast-01",
        "case": case_id,
        "at": "2026-11-13T21:00:00.000Z",
        "horizon_minutes": horizon,
        "quantiles": list(entry.QUANTILES),
        "event": {"slug": "next-ceo-of-nvidia", "title": "Next CEO of NVIDIA?"},
        "markets": [
            {"id": mid, "name": mid, "slug": mid, "yes_token": "1",
             "start": "2026-11-13T20:58:00.000Z", "prices": prices}
            for mid, prices in markets.items()
        ],
    }


def walk(n: int, start: float = 0.3, tick: float = 0.0, every: int = 1) -> list[float]:
    """A deterministic price series: flat, with a tick every `every` minutes."""
    out = []
    x = start
    for i in range(n):
        if tick and i % every == 0:
            x = min(1.0, max(0.0, x + (tick if (i // every) % 2 == 0 else -tick)))
        out.append(round(x, 6))
    return out


class ForecastShape(unittest.TestCase):
    def test_each_market_once_with_horizon_rows_of_five_ordered_quantiles(self):
        case = make_case({"a": walk(60, 0.31, 0.005, 3), "b": walk(10, 0.02), "c": [0.5]})
        out = entry.forecast_case(case)
        self.assertEqual(list(out), ["forecasts"])
        self.assertEqual(sorted(r["id"] for r in out["forecasts"]), ["a", "b", "c"])
        for row in out["forecasts"]:
            self.assertEqual(list(row), ["id", "q"])
            self.assertEqual(len(row["q"]), 240)
            for r in row["q"]:
                self.assertEqual(len(r), 5)
                self.assertEqual(r, sorted(r), "quantiles must not cross")
                for v in r:
                    self.assertTrue(0.0 <= v <= 1.0)
                    self.assertEqual(v, round(v, 6))

    def test_median_starts_at_the_last_price_and_fans_out(self):
        case = make_case({"a": walk(240, 0.31, 0.004, 2)})
        rows = entry.forecast_case(case)["forecasts"][0]["q"]
        self.assertAlmostEqual(rows[0][2], 0.31, places=6)
        first = rows[0][4] - rows[0][0]
        last = rows[-1][4] - rows[-1][0]
        self.assertGreater(last, first)

    def test_forecast_stays_inside_zero_and_one(self):
        case = make_case({"low": walk(500, 0.004, 0.02, 1), "high": walk(500, 0.995, 0.02, 2)})
        for row in entry.forecast_case(case)["forecasts"]:
            for r in row["q"]:
                self.assertGreaterEqual(r[0], 0.0)
                self.assertLessEqual(r[4], 1.0)

    def test_a_quiet_market_is_forecast_tighter_than_a_jumpy_one(self):
        case = make_case({"quiet": walk(600, 0.3, 0.0005, 120), "jumpy": walk(600, 0.3, 0.02, 7)})
        rows = {r["id"]: r["q"] for r in entry.forecast_case(case)["forecasts"]}
        quiet = rows["quiet"][-1][4] - rows["quiet"][-1][0]
        jumpy = rows["jumpy"][-1][4] - rows["jumpy"][-1][0]
        self.assertLess(quiet, jumpy)

    def test_a_short_history_still_forecasts(self):
        case = make_case({"new": [0.07]})
        rows = entry.forecast_case(case)["forecasts"][0]["q"]
        self.assertEqual(len(rows), 240)
        self.assertAlmostEqual(rows[-1][2], 0.07, places=6)

    def test_the_run_is_byte_for_byte_deterministic(self):
        case = make_case({"a": walk(400, 0.2, 0.01, 5), "b": walk(400, 0.03, 0.002, 9)})
        first = json.dumps(entry.forecast_case(case), separators=(",", ":"))
        second = json.dumps(entry.forecast_case(case), separators=(",", ":"))
        self.assertEqual(first, second)

    def test_two_process_runs_write_the_same_bytes_and_fit_the_size_cap(self):
        markets = {f"m{i}": walk(4320, 0.05 + i * 0.01, 0.002 * (i + 1), 11 + i) for i in range(24)}
        case = make_case(markets)
        blobs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                in_dir, out_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
                os.makedirs(in_dir)
                with open(os.path.join(in_dir, "t00.in"), "w", encoding="utf-8") as fh:
                    json.dump(case, fh, separators=(",", ":"), sort_keys=True)
                env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir, MAKEFUN_OUTPUT_DIR=out_dir)
                proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                                      env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                with open(os.path.join(out_dir, "t00.out"), "rb") as fh:
                    blobs.append(fh.read())
        self.assertEqual(blobs[0], blobs[1])
        self.assertLess(len(blobs[0]), 1048576)
        payload = json.loads(blobs[0])
        backtest.validate(case, payload)

    def test_other_quantile_levels_are_refused(self):
        case = make_case({"a": [0.3, 0.31]})
        case["quantiles"] = [0.1, 0.3, 0.5, 0.7, 0.9]
        with self.assertRaises(ValueError):
            entry.forecast_case(case)

    def test_a_bare_run_with_no_case_directory_exits_zero(self):
        """The hand-off check runs `python main.py` with nothing set up."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if k not in ("MAKEFUN_INPUT_DIR", "MAKEFUN_OUTPUT_DIR")}
            proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                                  env=env, cwd=tmp, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

    def test_a_missing_case_directory_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no-such-dir")
            out = os.path.join(tmp, "out")
            proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), missing, out],
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("no case directory", proc.stderr)
            self.assertFalse(os.path.exists(out))

    def test_an_empty_case_directory_writes_nothing_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir, out_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(in_dir)
            proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), in_dir, out_dir],
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(os.listdir(out_dir), [])

    def test_arguments_beat_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir, out_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(in_dir)
            with open(os.path.join(in_dir, "x00.in"), "w", encoding="utf-8") as fh:
                json.dump(make_case({"a": [0.3, 0.31, 0.32]}, "x00"), fh)
            env = dict(os.environ, MAKEFUN_INPUT_DIR=os.path.join(tmp, "unused"))
            proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), in_dir, out_dir],
                                  env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(os.listdir(out_dir), ["x00.out"])
            self.assertFalse(os.path.exists(os.path.join(tmp, "unused")))

    def test_find_output_dir_falls_back_beside_the_script(self):
        saved = os.environ.pop("MAKEFUN_OUTPUT_DIR", None)
        try:
            if not os.path.isdir("/output"):
                self.assertEqual(entry.find_output_dir(), os.path.join(ROOT, "out"))
            self.assertEqual(entry.find_output_dir("somewhere/else"), "somewhere/else")
        finally:
            if saved is not None:
                os.environ["MAKEFUN_OUTPUT_DIR"] = saved


class Scoring(unittest.TestCase):
    def test_pinball_matches_the_rules_worked_example(self):
        loss, median_error, covered = backtest.pinball(RULES_EXAMPLE_ROWS, [0.30, 0.35])
        self.assertAlmostEqual(loss, 1.01, places=6)
        self.assertAlmostEqual(median_error, 2.5, places=6)
        self.assertEqual(covered, 1)

    def test_baseline_matches_the_rules_example_rows(self):
        rows = backtest.baseline_rows(0.295)
        self.assertEqual(rows[0], [0.290888, 0.293314, 0.295, 0.296686, 0.299112])
        self.assertEqual(rows[-1], [0.231295, 0.268877, 0.295, 0.321123, 0.358705])

    def test_invalid_case_scores_the_flat_forecast(self):
        flat = [[0.5] * 5] * 2
        loss, _, _ = backtest.pinball(flat, [0.30, 0.35])
        self.assertAlmostEqual(loss, 8.75, places=6)

    def test_the_kit_stitches_and_leaves_p05_unscored(self):
        kit = backtest.kit_input_dir(None)
        cases = backtest.load_cases(kit)
        self.assertEqual(sorted(cases), ["p00", "p01", "p02", "p03", "p04", "p05"])
        timelines = backtest.timelines(cases)
        backtest.check_consistency(cases, timelines)
        self.assertIsNotNone(backtest.actuals(cases["p04"], timelines))
        self.assertIsNone(backtest.actuals(cases["p05"], timelines))

    def test_truncated_case_keeps_only_the_last_72_hours(self):
        prices = list(range(5000))
        case = make_case({"a": prices})
        short = backtest.truncated_case(case)
        self.assertEqual(len(short["markets"][0]["prices"]), 72 * 60)
        self.assertEqual(short["markets"][0]["prices"][-1], 4999)


if __name__ == "__main__":
    unittest.main()
