# SWE1 — swe1_20260907_015146.json   n=20   (full gold coverage: 9)

## Regime

- surface / window: min 2.00x, median 3.94x, max 5.86x
- instances in target regime (>1x): 20 of 20
- slices: 148 total, 4-12 per instance
- files split by a cut: 82; symbols straddling a cut: 122
- gold coverage by the surface: {'full': 9, 'partial': 9, 'none': 2}

## All instances  (n=20)

| metric | TRUNC | SOLO | KAN |
|---|---:|---:|---:|
| file_recall | 0.192 | 0.275 | 0.358 |
| file_precision | 0.350 | 0.350 | 0.525 |
| hunk_line_recall | 0.091 | 0.083 | 0.089 |
| normcontrol passed | 0.350 | 0.350 | 0.550 |
| non-empty patch | 0.350 | 0.350 | 0.550 |
| edits raw | 1.05 | 0.95 | 1.35 |
| edits kept | 0.75 | 0.65 | 0.80 |
| VDP survival | 0.290 | 0.300 | 0.414 |
| MDP unresolved | 0.45 | 0.05 | 0.50 |
| slices read | 1.0 | 7.4 | 7.4 |
| chars read | 29905 | 208263 | 208263 |
| calls | 1.0 | 7.4 | 7.4 |
| prompt tokens | 7798 | 54904 | 53723 |
| completion tokens | 430 | 688 | 932 |
| declined (NO EDITS) | 0.6 | 6.8 | 6.2 |
| truncated answers | 0.00 | 0.00 | 0.05 |
| wall minutes | 3.4 | 16.2 | 18.8 |

## Subset with full gold coverage  (n=9)

| metric | TRUNC | SOLO | KAN |
|---|---:|---:|---:|
| file_recall | 0.296 | 0.519 | 0.556 |
| file_precision | 0.444 | 0.556 | 0.667 |
| hunk_line_recall | 0.097 | 0.166 | 0.121 |
| normcontrol passed | 0.444 | 0.556 | 0.667 |
| non-empty patch | 0.444 | 0.556 | 0.667 |
| edits raw | 1.22 | 1.89 | 1.22 |
| edits kept | 1.00 | 1.22 | 1.00 |
| VDP survival | 0.422 | 0.444 | 0.556 |
| MDP unresolved | 0.44 | 0.11 | 0.67 |
| slices read | 1.0 | 8.2 | 8.2 |
| chars read | 29911 | 231996 | 231996 |
| calls | 1.0 | 8.2 | 8.2 |
| prompt tokens | 7801 | 61243 | 59949 |
| completion tokens | 491 | 973 | 1212 |
| declined (NO EDITS) | 0.4 | 7.0 | 6.9 |
| truncated answers | 0.00 | 0.00 | 0.11 |
| wall minutes | 3.5 | 19.0 | 21.6 |

## Pre-registered thresholds

| code | quantity | measured | prediction | falsification | verdict |
|---|---|---:|---|---|---|
| **SWE1-decomp** | file_recall(SOLO) - file_recall(TRUNC) | **+0.083** | >= +0.15 | <= 0 | not confirmed |
| **SWE1-kan** | file_recall(KAN) - file_recall(SOLO) | **+0.083** | > 0 | <= 0 | CONFIRMED |
| **SWE1-hunk** | hunk_recall(KAN) - hunk_recall(SOLO) | **+0.006** | > 0 | <= 0 | CONFIRMED |
| **SWE1-norm** | normcontrol rate, KAN | **+0.550** | >= 0.70 | < 0.50 | not confirmed |
| **SWE1-vdp** | VDP survival, KAN | **+0.414** | >= 0.60 | < 0.35 | not confirmed |
| **SWE1-mdp** | MDP unresolved, KAN - SOLO | **+0.450** | <= 0 | > 0 | FALSIFIED |
| **SWE1-floor** | file_recall TRUNC | **+0.192** | - | < 0.15 -> out of range | CONFIRMED |

## Paired comparison (per instance, same surface, same slices)

| comparison | metric | wins | losses | ties | sign-test p |
|---|---|---:|---:|---:|---:|
| KAN vs SOLO | file_recall | 5 | 2 | 13 | 0.453 |
| KAN vs SOLO | hunk_recall | 5 | 4 | 11 | 1.000 |
| SOLO vs TRUNC | file_recall | 4 | 2 | 14 | 0.688 |
| SOLO vs TRUNC | hunk_recall | 4 | 5 | 11 | 1.000 |

## VDP rejections

| reason | TRUNC | SOLO | KAN |
|---|---:|---:|---:|
| `no_op` | 5 | 4 | 8 |
| `not_found` | 1 | 2 | 3 |

## Per instance

| repo | ratio | slices | gold | in surf | TRUNC f/h | SOLO f/h | KAN f/h |
|---|---:|---:|---:|---:|---|---|---|
| ansible | 4.34x | 9 | 1 | 1 | 0.00/0.00 | 1.00/0.61 | 1.00/0.06 |
| openlibrary | 2.00x | 4 | 2 | 1 | 0.00/0.00 | 0.00/0.00 | 0.50/0.50 |
| qutebrowser | 3.94x | 8 | 2 | 1 | 0.00/0.00 | 0.00/0.00 | 0.50/0.04 |
| ansible | 3.27x | 7 | 1 | 1 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| openlibrary | 2.79x | 6 | 2 | 2 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| qutebrowser | 4.22x | 9 | 2 | 1 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| ansible | 4.03x | 8 | 1 | 1 | 0.00/0.00 | 1.00/0.37 | 1.00/0.21 |
| openlibrary | 2.28x | 5 | 2 | 1 | 0.50/0.37 | 0.50/0.09 | 0.00/0.00 |
| qutebrowser | 4.21x | 9 | 3 | 2 | 0.33/0.24 | 0.00/0.00 | 0.00/0.00 |
| ansible | 4.39x | 9 | 1 | 1 | 1.00/0.18 | 0.00/0.00 | 1.00/0.18 |
| openlibrary | 2.26x | 5 | 2 | 1 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| qutebrowser | 3.66x | 8 | 3 | 3 | 0.33/0.00 | 1.00/0.23 | 0.33/0.07 |
| ansible | 3.19x | 7 | 2 | 0 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| openlibrary | 2.30x | 5 | 3 | 2 | 0.33/0.35 | 0.33/0.07 | 0.33/0.07 |
| qutebrowser | 5.86x | 12 | 2 | 1 | 0.00/0.00 | 0.00/0.00 | 0.50/0.00 |
| ansible | 4.38x | 9 | 3 | 3 | 0.33/0.09 | 0.67/0.26 | 0.67/0.26 |
| openlibrary | 2.52x | 5 | 3 | 0 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| qutebrowser | 4.17x | 9 | 2 | 2 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| ansible | 4.38x | 9 | 1 | 1 | 1.00/0.60 | 1.00/0.02 | 1.00/0.31 |
| openlibrary | 2.44x | 5 | 3 | 2 | 0.00/0.00 | 0.00/0.00 | 0.33/0.08 |

**12.95 h for 20 instances = 38.8 min/instance (three arms)**
