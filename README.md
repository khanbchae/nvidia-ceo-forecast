# nvidia-ceo-forecast

An entry for make.fun's `polymarket-nvidia-ceo-forecast-01` bounty: forecast the
Yes price of every market of Polymarket's "Next CEO of NVIDIA?" event at every
minute of the four hours after a checkpoint, as five quantiles.

* `main.py` — the entry point. Python 3.13, standard library only, no network.
  Reads `$MAKEFUN_INPUT_DIR/*.in` (default `/input`), writes
  `$MAKEFUN_OUTPUT_DIR/<id>.out` (default `/output`). Deterministic.
* `backtest.py` — scores the forecast on the kit's practice cases against the
  house baseline, with the contest's pinball loss.
* `tests/` — `python3 -m unittest discover -s tests`.

## The model

A driftless walk in price space whose step size is measured from the market's
own recent prices:

    q_tau(h) = clamp(x + z_tau * s * sqrt(h), 0, 1)

`s` is the mean absolute one-minute move over the last 3 hours blended half and
half with the last 24 hours, converted from a mean absolute move to a standard
deviation (`sqrt(pi/2)`) and scaled by 0.45. `z_tau` is the tau-quantile of a
two-component mixture: 87.5% N(0, 1) and 12.5% N(0, 10.5^2) — a quiet core with
a rare wide jump, which is what these order books do (a market sits still for
minutes, then ticks several cents at once).

## Running it

    # one case in, one forecast out, the way the grader does it
    mkdir -p out
    MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py

    # or, with no environment at all: case dir first, output dir second
    python3 main.py <kit>/input out

A bare `python3 main.py`, with no arguments and no environment, looks in
`$MAKEFUN_INPUT_DIR`, then `/input`, then `input/` beside the file; when none of
them exists it prints where it looked and exits 0 instead of raising.

    # score it on the practice cases (5 of 6; the kit holds nothing after
    # p05's checkpoint)
    python3 backtest.py --detail --per-market
    python3 backtest.py --subprocess        # run main.py as the grader does

On a machine where `python3` is not on PATH (this one), use `py -3` in place of
`python3`.

`backtest.py` finds the kit at `$MAKEFUN_KIT` or `--kit`, defaulting to the path
this repository was built from.

## What the practice says

Mean pinball loss in points, lower is better, over the five practice cases the
kit can answer:

| version | model | loss |
| --- | --- | --- |
| house baseline | fixed spread 0.0025 per sqrt(min) | 0.396429 |
| v1 | per-market realized volatility, 6h window, normal quantiles | 0.394995 |
| v2 | mean absolute move over 6h, scale 0.5, jump-mixture quantiles | 0.323000 |
| v3 | 3h/24h blend, scale 0.45, same mixture | **0.315310** |

v3 leaves-one-case-out at 0.319 (each case's parameters fitted on the other
four), so the gain is not one window's luck. Its 90% coverage on the practice is
0.910, near the 0.9 an honest forecast should get.

## License

MIT, see `LICENSE`.
