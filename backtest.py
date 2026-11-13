#!/usr/bin/env python3
"""Backtest the forecast in main.py against the kit's practice cases.

The kit holds six case files (p00..p05), each the prices up to that case's
checkpoint. Stitching them together gives one timeline per market, so a case's
next 240 minutes are the later case files' own prices: a forecast made at a
checkpoint is scored against prices the case file does not contain. p05's
checkpoint is the last minute the kit holds, so it has no answer and is not
scored.

Usage (from the repository root):

    python3 backtest.py                 # in-process, every scored case
    python3 backtest.py --subprocess    # run main.py the way the grader does
    python3 backtest.py --cases p00,p03 # a subset
    python3 backtest.py --per-market    # per market loss table

The reported loss is the contest's: the mean pinball loss in percentage points
over every case, market, scored minute and quantile level. Lower is better.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import main as forecast_main

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
HORIZON = 240
BASELINE_SIGMA = 0.0025

DEFAULT_KIT = r"C:\Users\Admin\Desktop\makefun\polymarket-nvidia-ceo-forecast-01-kit"


def kit_input_dir(explicit: str | None) -> str:
    path = explicit or os.environ.get("MAKEFUN_KIT") or DEFAULT_KIT
    path = os.path.join(path, "input")
    if not os.path.isdir(path):
        raise SystemExit(f"no kit input directory at {path}; pass --kit")
    return path


def load_cases(input_dir: str) -> dict[str, dict]:
    cases = {}
    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".in"):
            with open(os.path.join(input_dir, name), encoding="utf-8") as fh:
                case = json.load(fh)
            cases[case["case"]] = case
    return cases


def timelines(cases: dict[str, dict]) -> dict[str, list[float]]:
    """One price series per market, the longest the kit holds."""
    out: dict[str, list[float]] = {}
    for case in cases.values():
        for m in case["markets"]:
            prev = out.get(m["id"])
            if prev is None or len(m["prices"]) > len(prev):
                out[m["id"]] = m["prices"]
    return out


def check_consistency(cases: dict[str, dict], tl: dict[str, list[float]]) -> None:
    """Every case's prices must be a prefix of the stitched timeline."""
    for case in cases.values():
        for m in case["markets"]:
            full = tl[m["id"]]
            if full[: len(m["prices"])] != m["prices"]:
                raise SystemExit(f"market {m['name']} differs between case files")


def truncated_case(case: dict) -> dict:
    """The case as the grader hands it over: 72 hours up to the checkpoint."""
    markets = []
    for m in case["markets"]:
        prices = m["prices"][-min(len(m["prices"]), 72 * 60):]
        markets.append({"id": m["id"], "name": m["name"], "slug": m["slug"],
                        "yes_token": m["yes_token"], "start": m["start"], "prices": prices})
    return {"schema": case["schema"], "case": case["case"], "at": case["at"],
            "horizon_minutes": case["horizon_minutes"], "quantiles": case["quantiles"],
            "event": case["event"], "markets": markets}


def actuals(case: dict, tl: dict[str, list[float]]) -> dict[str, list[float]] | None:
    """The 240 prices after the checkpoint for each market, or None if unheld."""
    out = {}
    n = len(case["markets"][0]["prices"])
    for m in case["markets"]:
        full = tl[m["id"]]
        n = len(m["prices"])
        if len(full) < n - 1 + HORIZON:
            return None
        out[m["id"]] = full[n - 1: n - 1 + HORIZON]
    return out


def pinball(rows: list[list[float]], ys: list[float]) -> tuple[float, float, int]:
    """Mean pinball loss in points, mean absolute median error, covered minutes."""
    total = 0.0
    med = 0.0
    cover = 0
    for qs, y in zip(rows, ys):
        for tau, q in zip(QUANTILES, qs):
            d = y - q
            total += max(tau * d, (tau - 1.0) * d) * 100.0
        med += abs(y - qs[2]) * 100.0
        if qs[0] <= y <= qs[4]:
            cover += 1
    return total / (len(rows) * len(QUANTILES)), med / len(rows), cover


def baseline_rows(x: float, horizon: int = HORIZON) -> list[list[float]]:
    """The house baseline: persistence with a fixed random-walk spread."""
    return [[forecast_main.price6(x + z * BASELINE_SIGMA * (h ** 0.5)) for z in Z]
            for h in range(1, horizon + 1)]


def run_entry(case: dict, root: str, timeout: float = 8.0) -> tuple[dict, float]:
    """Run main.py as the grader does: one case, its own input and output dirs."""
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "input")
        out_dir = os.path.join(tmp, "output")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        with open(os.path.join(in_dir, case["case"] + ".in"), "w", encoding="utf-8") as fh:
            json.dump(case, fh, separators=(",", ":"), sort_keys=True)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir, MAKEFUN_OUTPUT_DIR=out_dir)
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, os.path.join(root, "main.py")],
                              env=env, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise SystemExit(f"main.py failed on {case['case']}:\n{proc.stdout}\n{proc.stderr}")
        with open(os.path.join(out_dir, case["case"] + ".out"), encoding="utf-8") as fh:
            return json.load(fh), elapsed


def validate(case: dict, out: dict) -> None:
    """The output shape the contest requires."""
    if set(out) != {"forecasts"}:
        raise SystemExit(f"output keys {sorted(out)}")
    ids = [row["id"] for row in out["forecasts"]]
    if sorted(ids) != sorted(m["id"] for m in case["markets"]):
        raise SystemExit("output does not hold each market exactly once")
    for row in out["forecasts"]:
        if set(row) != {"id", "q"}:
            raise SystemExit(f"row keys {sorted(row)}")
        q = row["q"]
        if len(q) != case["horizon_minutes"]:
            raise SystemExit(f"{len(q)} rows, want {case['horizon_minutes']}")
        for r in q:
            if len(r) != len(QUANTILES):
                raise SystemExit(f"row of {len(r)} quantiles")
            for v in r:
                if not (0.0 <= v <= 1.0) or v != v:
                    raise SystemExit(f"quantile {v} out of range")
            if any(r[i] > r[i + 1] for i in range(len(r) - 1)):
                raise SystemExit("quantiles not non-decreasing")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=None)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--subprocess", action="store_true")
    ap.add_argument("--per-market", action="store_true")
    ap.add_argument("--detail", action="store_true", help="print each case's loss")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.abspath(__file__))
    cases = load_cases(kit_input_dir(args.kit))
    tl = timelines(cases)
    check_consistency(cases, tl)
    wanted = dict.fromkeys(args.cases.split(",")) if args.cases else None

    losses: list[float] = []
    base_losses: list[float] = []
    per_market: dict[str, list[float]] = {}
    per_market_base: dict[str, list[float]] = {}
    med_err: list[float] = []
    cover: list[float] = []
    times: list[float] = []
    skipped: list[str] = []

    for case_id in sorted(cases):
        if wanted is not None and case_id not in wanted:
            continue
        case = cases[case_id]
        act = actuals(case, tl)
        if act is None:
            skipped.append(case_id)
            continue
        short = truncated_case(case)
        if args.subprocess:
            out, elapsed = run_entry(short, root)
            validate(short, out)
            times.append(elapsed)
            rows = {row["id"]: row["q"] for row in out["forecasts"]}
        else:
            rows = {m["id"]: forecast_main.forecast_market(m["prices"], HORIZON)
                    for m in short["markets"]}
        case_total = 0.0
        case_base = 0.0
        case_n = 0
        for m in short["markets"]:
            ys = act[m["id"]]
            loss, mae, cov = pinball(rows[m["id"]], ys)
            x = m["prices"][-1]
            bloss, _, _ = pinball(baseline_rows(x), ys)
            per_market.setdefault(m["name"], []).extend([loss])
            per_market_base.setdefault(m["name"], []).extend([bloss])
            case_total += loss
            case_base += bloss
            case_n += 1
            med_err.append(mae)
            cover.append(cov / HORIZON)
        losses.append(case_total / case_n)
        base_losses.append(case_base / case_n)
        if args.detail:
            print(f"  {case_id}: loss {case_total / case_n:.6f}   baseline {case_base / case_n:.6f}")

    if not losses:
        raise SystemExit("no case could be scored")

    loss = sum(losses) / len(losses)
    base = sum(base_losses) / len(base_losses)
    print(f"cases scored: {len(losses)}" + (f"  (no answer in the kit: {', '.join(skipped)})" if skipped else ""))
    print(f"loss           {loss:.6f}")
    print(f"house baseline {base:.6f}")
    print(f"delta          {loss - base:+.6f}")
    print(f"median error   {sum(med_err) / len(med_err):.6f}")
    print(f"90% coverage   {sum(cover) / len(cover):.6f}")
    if times:
        print(f"run seconds    max {max(times):.2f}  mean {sum(times) / len(times):.2f}")
    if args.per_market:
        print()
        print(f"{'market':28s} {'loss':>9s} {'baseline':>9s} {'delta':>9s}")
        for name in sorted(per_market, key=lambda n: -(sum(per_market[n]) / len(per_market[n]))):
            a = sum(per_market[name]) / len(per_market[name])
            b = sum(per_market_base[name]) / len(per_market_base[name])
            print(f"{name[:28]:28s} {a:9.6f} {b:9.6f} {a - b:+9.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
