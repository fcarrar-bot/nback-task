# Visual Letter N-Back Task

PsychoPy implementation for working memory assessment.

## Two Versions

| File | Duration | Trials | Use |
|------|----------|--------|-----|
| `nback_demo.py` | ~2-3 min | 15 | Quick test |
| `nback_full.py` | ~15 min | 96 (48×2) | Real data collection |

Both use **2-back** (research standard for individual differences).

## Quick Start

```bash
# Install dependencies
pip install psychopy pandas scipy

# Run demo
python nback_demo.py

# Run full version
python nback_full.py
```

Just double-click the .py file if PsychoPy is installed.

## Task

Press **SPACE** when the current letter matches the one from **2 trials ago**.

```
A ... B ... A → press SPACE on 2nd A
```

## Output

Data saved to `data/` folder:
- `{id}_demo_{timestamp}.csv` — Demo results
- `{id}_s{n}_trials_{timestamp}.csv` — Trial-by-trial (full)
- `{id}_s{n}_summary_{timestamp}.csv` — Summary (full)
- `all_participants.csv` — Aggregated (full)

## Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` | % correct responses |
| `dprime` | d' sensitivity (main measure) |
| `hits` | Correct target detections |
| `misses` | Missed targets |
| `false_alarms` | Wrong presses |
| `mean_rt_ms` | Reaction time |

## Exclusion Criteria

Exclude participants with:
- **d' < 1.1** (>2 SD below mean)
- **Accuracy < 60%**

## Requirements

- Python 3.8+
- PsychoPy
- pandas
- scipy

## License

MIT
