# Visual Letter N-Back Task

A PsychoPy implementation for assessing working memory capacity.

## Overview

The N-back task is a standard cognitive psychology paradigm for measuring working memory. Participants monitor a sequence of stimuli and respond when the current stimulus matches one presented N trials earlier.

This implementation includes:
- **0-back** (control): Respond when target letter 'X' appears
- **2-back** (standard): Respond when current letter matches 2 trials back
- **3-back** (harder): Respond when current letter matches 3 trials back

### Two Modes

| Mode | Duration | Trials/block | Blocks | Conditions |
|------|----------|--------------|--------|------------|
| **Demo** | ~5 min | 20 | 1 | 2-back, 3-back |
| **Full** | ~20 min | 48 | 2 | 0-back, 2-back, 3-back |

## Requirements

- Python 3.8+
- PsychoPy 2023.1.0+
- pandas
- numpy
- scipy

### Installation

```bash
# Install PsychoPy (recommended: use standalone installer)
# https://www.psychopy.org/download.html

# Or via pip:
pip install psychopy pandas numpy scipy
```

## Usage

### Run the experiment

```bash
python nback_task.py
```

A dialog will prompt for:
- **Participant ID**: Unique identifier
- **Session**: Session number (default: 1)

### Task Structure

**Demo Mode (~5 min):**
1. 2-back: Practice (5) + Main (20 trials)
2. 3-back: Practice (5) + Main (20 trials)

**Full Mode (~20 min):**
1. 0-back: Practice (10) + 2 blocks × 48 trials
2. 2-back: Practice (10) + 2 blocks × 48 trials
3. 3-back: Practice (10) + 2 blocks × 48 trials

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Stimulus duration | 500ms | Letter display time |
| ISI | 2000ms | Inter-stimulus interval |
| Trials per block | 48 | Main task trials |
| Blocks per level | 2 | Repetitions per N-level |
| Target ratio | 30% | Proportion of targets |
| Response key | SPACE | Target detection |

## Output

### Individual Files

Each participant generates:
- `{id}_session{n}_trials_{timestamp}.csv` — Trial-by-trial data
- `{id}_session{n}_summary_{timestamp}.csv` — Condition summaries

### Aggregated File

All participants are appended to:
- `data/all_participants_summary.csv`

### Metrics

| Metric | Description |
|--------|-------------|
| `hits` | Correct target detections |
| `misses` | Missed targets (omission errors) |
| `false_alarms` | Responses to non-targets (commission errors) |
| `correct_rejections` | Correct non-responses |
| `accuracy` | (hits + correct_rejections) / total |
| `hit_rate` | hits / n_targets |
| `false_alarm_rate` | false_alarms / n_nontargets |
| `dprime` | d' = z(hit_rate) - z(fa_rate) — sensitivity |
| `criterion` | c = response bias |
| `mean_rt_hits` | Average RT for correct detections (ms) |

## Normative Data & Exclusion Criteria

Based on published research (healthy adults, 2-back task):

| Metric | Population Mean | SD | Exclusion (>2 SD) |
|--------|-----------------|----|--------------------|
| d' | ~2.5 | ~0.7 | < 1.1 |
| Accuracy | ~80% | ~10% | < 60% |
| Hit rate | ~85% | ~12% | < 61% |

**Recommendation:** Exclude participants with 2-back d' < 1.1 or accuracy < 60%

## Customization

Edit `CONFIG` dictionary in `nback_task.py`:

```python
CONFIG = {
    'stimulus_duration': 0.5,    # seconds
    'isi_duration': 2.0,         # seconds
    'n_levels': [0, 2],          # which N-back levels to run
    'trials_per_block': 48,
    'blocks_per_level': 2,
    'target_ratio': 0.30,
    'letters': ['A', 'B', ...],  # stimulus set
    'response_key': 'space',
    # ...
}
```

## References

- Jaeggi, S. M., et al. (2010). The relationship between n-back performance and matrix reasoning. Intelligence.
- Kane, M. J., et al. (2007). Working memory, attention control, and the n-back task. Journal of Experimental Psychology.
- Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d′. Behavior Research Methods.

## License

MIT
