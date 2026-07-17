# Isolated Strategy Lab

This is a new, independent Git project for five research variants. It does not
share the production repository's Git history or remote.

## Frozen production baseline

- Source repository (renamed): `https://github.com/appr1ciat1/tst_wocker`
- Source commit: `d0fe61b0a1701873868b301e52cf2820de28bf84`
- Source-only baseline snapshot commit: `509e06d`
- Immutable local tag: `baseline-original-d0fe61b`

The source commit is the 2026-07-15 automated report refresh and predates all
five variants. The baseline commit contains only strategy source, configuration
and tests; bulky generated production reports are intentionally excluded.

The lab is published only to
`https://github.com/appr1ciat1/tw_stocker_filter_lab`. A local pre-push hook
blocks both the old production URL and its current URL
`https://github.com/appr1ciat1/tst_wocker.git`.

## Five variants

| Parent | Purpose | Registry name |
|---|---|---|
| v8.5 | Overnight/global-leader/chip confirmation | `momentum_v85_confirmed` |
| v8.5 | TWD 300,000 allocation | `momentum_v85_300k` |
| SURGE PRO | Overnight/global-leader/chip confirmation | `mom_surge_pro_confirmed` |
| SURGE PRO | TWD 300,000 allocation | `mom_surge_pro_300k` |
| SURGE PRO | Capital-rotation warning only | `mom_surge_pro_rotation_alert` |

## Comparison rule

Each candidate must be compared with its parent using the same frozen data,
universe, costs, slippage, dates and execution timing. Promotion requires
out-of-sample improvement in return/drawdown trade-off, stable subperiods and
no look-ahead leakage.

```bash
git diff --stat baseline-original-d0fe61b..main
git show baseline-original-d0fe61b:strategies/optimized_v85.py
python -m twstk.backtest.runner --list
pytest -q test_new_strategy_variants.py
```

The public destination for this isolated project is
`appr1ciat1/tw_stocker_filter_lab`. Add only that URL as `origin`; the
production repository remains blocked by the local pre-push guard.
