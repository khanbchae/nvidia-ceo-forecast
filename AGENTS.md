# nvidia-ceo-forecast

Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`: Python 3.13, standard
library only, `main.py` at the repository root (the contest's entry point) plus a
`backtest.py` that scores it on the kit's practice cases.

## Status

Updated 2026-11-13. Works: `main.py` writes a valid forecast for each case in
~0.05s and 251 KB (limit 8s and 1 MiB), byte-identical across runs; practice
loss 0.315310 against the house baseline's 0.396429 over the five cases the kit
can answer; 18 tests pass. A bare `main.py` with no arguments and no environment
finds no case directory on a Windows box and must exit 0 with a message, not a
traceback (`/input` does not exist here). Open: p05 cannot be backtested (the
kit holds no prices after its checkpoint), so the practice number is a five-case
number, not the six-case one RULES quotes.

## Run

    py -3 -m unittest discover -s tests            # 18 tests
    py -3 backtest.py --detail --per-market        # practice score
    py -3 main.py <kit>/input out                  # case dir, output dir
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out py -3 main.py

`python3` is not on PATH on this machine; `py -3` is. The kit is at
`$MAKEFUN_KIT` (default: the build machine's
`~/Desktop/makefun/polymarket-nvidia-ceo-forecast-01-kit`); `backtest.py` reads
its `input/` only.

## Map

- `main.py` — read every `*.in`, write `*.out`. `forecast_market` is the model;
  `step_size` the volatility; `Z` the mixture quantiles.
- `backtest.py` — `load_cases`, `timelines` (one series per market, stitched
  from the six case files), `truncated_case` (72h, as the grader hands it over),
  `actuals`, `pinball`, `baseline_rows`, `run_entry`, `validate`.
- `tests/test_forecast.py` — shape, bounds, determinism, size cap, the RULES
  worked example (loss 1.01, median error 2.5, coverage 0.5), and the baseline's
  quoted rows for x=0.295.

## Decisions

- Score with `backtest.py` in-process while tuning (fast); confirm with
  `--subprocess`, which runs `main.py` in the grader's shape.
- Volatility: mean absolute move beats RMS and MAD (RMS over 24h is skewed by
  early listing jumps, MAD is 0 whenever a price did not tick in half the
  minutes). A 3h/24h blend beats either window alone.
- Shape: a normal's quantiles pay too much for the 25%/75% levels; the
  0.875/0.125 mixture with JUMP_K 10.5 wins on every shape tried (free pooled
  quantiles, per-horizon-band quantiles, horizon-growing jump probability).
- Tried and dropped: mean reversion to an EMA, momentum in the median, a
  case-wide "news" multiplier, a bend on the sigma -> spread map, per-market
  jump intensity, logit-space walks, smoothed centres, mid-lag volatility.
- p05's 240 minutes are not in the kit; do not report a six-case number.
- The entry point must survive a bare run on a machine with no `/input`: it
  resolves the case directory from an argument, `$MAKEFUN_INPUT_DIR`, `/input`,
  then `input/` beside itself, and exits 0 with a message when none exists.
