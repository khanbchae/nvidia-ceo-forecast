#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices 240 minutes ahead.

Reads every `<id>.in` case in $MAKEFUN_INPUT_DIR (default /input) and writes
`<id>.out` to $MAKEFUN_OUTPUT_DIR (default /output):

    {"forecasts": [{"id": <conditionId>, "q": [[5 quantiles] x 240 rows]}]}

Row h-1 is the forecast for the minute h minutes after the case checkpoint, and
the five numbers of a row are the quantiles at 0.05, 0.25, 0.5, 0.75 and 0.95.

Model
-----
The price is a driftless walk whose step size is estimated per market, and whose
one-minute innovations are a two-component mixture rather than a normal: most
minutes barely move, and now and then a tick or a jump moves the price several
steps' worth. Written out, for minute h after the checkpoint and the market's
last price x:

    q_tau(h) = clamp(x + z_tau * s * sqrt(h), 0, 1)

  * s is the market's own one-minute step size: the mean absolute one-minute
    move over the last `VOL_SHORT` minutes blended with the same over the last
    `VOL_LONG` minutes, scaled from the mean absolute move to a standard
    deviation of a normal (sqrt(pi/2) = 1.2533), then multiplied by `SCALE`.
    The blend keeps a market that just woke up from dominating, and keeps a
    market that jumped once in the last three hours from looking permanently
    volatile. The scale is below 1 because a heavy-tailed step needs a narrower
    core to sit at the same quartile.
  * z_tau is the tau-quantile of a mixture: with probability 1 - JUMP_P the
    innovation is N(0, 1), with probability JUMP_P it is N(0, JUMP_K^2). That
    reproduces what the practice shows: the 25% and 75% quantiles sit close to
    the median, while the 5% and 95% ones sit about three step-sizes out.

Everything is a function of the case's own prices, with no randomness, no
clock, no network and no other state, so two runs on one case write the same
bytes. The standard library is all it uses.

Directories: the first argument is the case directory and the second the output
directory; with no argument the case directory is $MAKEFUN_INPUT_DIR, else
/input (the grader's), else an `input` folder beside this file, and the output
directory is $MAKEFUN_OUTPUT_DIR, else /output, else `out` beside this file. With
no case directory anywhere it says where it looked and exits 0 rather than
raising, so a bare run on a machine that has not been set up is harmless.

Locally:
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
    python3 main.py <kit>/input out
"""

from __future__ import annotations

import json
import math
import os
import sys

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# Volatility estimation, in one-minute price units.
VOL_SHORT = 180           # minutes of history the short blend arm averages over
VOL_LONG = 1440           # minutes of history the long blend arm averages over
SHORT_WEIGHT = 0.5        # weight on the short arm
MEAN_ABS_TO_SD = math.sqrt(math.pi / 2.0)   # mean |N(0, sd)| -> sd
SCALE = 0.45              # step size -> core spread of the walk
MIN_STEP = 1e-6
MAX_STEP = 0.02

# Innovation shape: a quiet normal with a rare wide jump.
JUMP_P = 0.125
JUMP_K = 10.5


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mixture_quantiles(taus, p: float, k: float) -> tuple[float, ...]:
    """Quantiles of (1 - p) * N(0, 1) + p * N(0, k^2), by bisection on its CDF."""
    out = []
    for tau in taus:
        lo, hi = -60.0, 60.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if (1.0 - p) * _norm_cdf(mid) + p * _norm_cdf(mid / k) < tau:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return tuple(out)


Z = _mixture_quantiles(QUANTILES, JUMP_P, JUMP_K)


def price6(x: float) -> float:
    """x clamped to [0, 1] and rounded to 6 decimals (never -0.0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def mean_abs_move(prices: list[float], window: int) -> float | None:
    """Mean absolute one-minute move over the last `window` minutes, or None."""
    tail = prices[-(window + 1):]
    if len(tail) < 2:
        return None
    total = 0.0
    for i in range(1, len(tail)):
        total += abs(tail[i] - tail[i - 1])
    return total / (len(tail) - 1)


def step_size(prices: list[float]) -> float:
    """The market's one-minute step size, blended over two windows and scaled."""
    short = mean_abs_move(prices, VOL_SHORT)
    long = mean_abs_move(prices, VOL_LONG)
    if short is None and long is None:
        return 0.0
    if short is None:
        raw = long
    elif long is None:
        raw = short
    else:
        raw = SHORT_WEIGHT * short + (1.0 - SHORT_WEIGHT) * long
    step = SCALE * MEAN_ABS_TO_SD * raw
    return min(MAX_STEP, max(MIN_STEP, step))


def forecast_market(prices: list[float], horizon: int) -> list[list[float]]:
    """The horizon rows of five quantiles for one market."""
    x = prices[-1] if prices else 0.5
    step = step_size(prices)
    rows: list[list[float]] = []
    for h in range(1, horizon + 1):
        sd = step * math.sqrt(h)
        rows.append([price6(x + z * sd) for z in Z])
    return rows


def forecast_case(case: dict) -> dict:
    """The one output object for one case."""
    taus = list(case["quantiles"])
    if taus != list(QUANTILES):
        raise ValueError(f"case asks for quantiles {taus}")
    horizon = int(case["horizon_minutes"])
    return {"forecasts": [{"id": m["id"], "q": forecast_market(m["prices"], horizon)}
                          for m in case["markets"]]}


HERE = os.path.dirname(os.path.abspath(__file__))


def find_input_dir(override: str | None = None) -> str | None:
    """The case directory: an argument, then $MAKEFUN_INPUT_DIR, then /input,
    then an `input` folder beside this file. None when none of them exists."""
    tried: list[str] = []
    for cand in (override, os.environ.get("MAKEFUN_INPUT_DIR"), "/input",
                 os.path.join(HERE, "input")):
        if not cand:
            continue
        if os.path.isdir(cand):
            return cand
        tried.append(cand)
    print("main.py: no case directory, looked in " + ", ".join(tried)
          + "; pass one as the first argument or set MAKEFUN_INPUT_DIR", file=sys.stderr)
    return None


def find_output_dir(override: str | None = None) -> str:
    """The directory the forecasts go to: an argument, then $MAKEFUN_OUTPUT_DIR,
    then /output, then an `out` folder beside this file."""
    if override:
        return override
    env = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if env:
        return env
    if os.path.isdir("/output"):
        return "/output"
    return os.path.join(HERE, "out")


def main(argv: list[str]) -> int:
    input_dir = find_input_dir(argv[0] if argv else None)
    if input_dir is None:
        return 0
    output_dir = find_output_dir(argv[1] if len(argv) > 1 else None)
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(f for f in os.listdir(input_dir) if f.endswith(".in")):
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        body = json.dumps(forecast_case(case), separators=(",", ":"))
        with open(os.path.join(output_dir, case_id + ".out"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(body + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
