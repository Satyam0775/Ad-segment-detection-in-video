# EVAL.md — Evaluation

## 1. Test set

The five official inputs from the assignment brief (Section 4), used
exactly as specified, no substitutions:

| ID | Type | URL |
|---|---|---|
| `video1_longform_youtube` | Long-form YouTube | https://www.youtube.com/watch?v=ujFWRFYLGjY |
| `video2_youtube_short` | YouTube Short | https://www.youtube.com/shorts/Ve0zdhTQA4U |
| `video3_livestream_or_recording` | Live stream (news) | https://www.youtube.com/watch?v=s0LLVQeMmtU |
| `video4_instagram_reel_1` | Instagram Reel | https://www.instagram.com/reel/DJl6-v8oufg |
| `video5_instagram_reel_2` | Instagram Reel | https://www.instagram.com/reel/Db78RoIuOyl/ |

**Acquisition method per video** — *(fill in once run; template below)*:

| ID | Acquisition | Notes |
|---|---|---|
| `video1_longform_youtube` | yt-dlp, `--kind` auto | *PENDING — run `scripts/run_pipeline.py` and note actual result* |
| `video2_youtube_short` | yt-dlp, `--kind short` | *PENDING* |
| `video3_livestream_or_recording` | yt-dlp live capture, OR 20+ min recorded fallback | *PENDING — state which was used and why* |
| `video4_instagram_reel_1` | Manual capture → local file, per assignment Reels rule | *PENDING — state exact capture method used* |
| `video5_instagram_reel_2` | Manual capture → local file | *PENDING* |

## 2. Ground truth methodology

Hand-labeled by watching each video start to finish myself (see
`ground_truth/README.md` for the process and schema). Labels use the same
`ad_type` taxonomy as the detector output, and apply the rulings recorded
in `DESIGN.md` Section 4 consistently. **This file currently ships with a
template only (`ground_truth/ground_truth.template.json`) — the real
`ground_truth.json` is not included because inventing labels would defeat
the purpose of the exercise.**

## 3. Metrics

- **Segment-level IoU** (`evaluation/metrics.py::iou`): intersection over
  union of predicted vs. ground-truth time intervals, in [0,1].
- **Matching** (`match_segments`): greedy one-to-one match, each ground
  truth segment paired with its highest-IoU unclaimed prediction above a
  0.3 IoU threshold (deliberately lenient — we care first about "found the
  right region," and report boundary tightness separately).
- **Boundary error** (`boundary_error`): `|pred.start - gt.start|` and
  `|pred.end - gt.end|` in seconds, reported as a mean over true positives.
- **Precision / Recall / F1**: computed from the match result — unmatched
  predictions are false positives, unmatched ground truth are false
  negatives.

Why these over raw accuracy: raw "% of ads found" hides *where* a
prediction was wrong (badly placed boundary vs. missed entirely vs. spurious
detection), which is what actually matters for a timeline consumer (e.g. a
skip button). IoU + boundary error together answer both "did we find it"
and "how precisely."

## 4. Per-video results

**PENDING EXECUTION.** Run:
```cmd
python scripts\run_pipeline.py <url_or_path> --out artifacts\<video_id>.json
```
for each of the five videos, fill in `ground_truth/ground_truth.json`
yourself, then run:
```cmd
python scripts\run_evaluation.py
```
and paste the printed per-video table here. `run_evaluation.py` will report
any video as `PENDING` rather than silently fabricating a number if either
the prediction file or ground truth entry is missing — this table should
only ever contain numbers that came out of that script.

| Video ID | Precision | Recall | F1 | Mean IoU (TP) | Mean start err (s) | Mean end err (s) |
|---|---|---|---|---|---|---|
| video1_longform_youtube | PENDING | | | | | |
| video2_youtube_short | PENDING | | | | | |
| video3_livestream_or_recording | PENDING | | | | | |
| video4_instagram_reel_1 | PENDING | | | | | |
| video5_instagram_reel_2 | PENDING | | | | | |

## 5. Overall results

**PENDING** — mean F1 across evaluated videos, printed by
`scripts/run_evaluation.py`'s summary line.

## 6. Failure cases (minimum three required)

**PENDING EXECUTION.** `evaluation/failure_analysis.py` provides the
structure (`FailureCase` dataclass: expected / predicted / error /
root cause / failed signal / proposed improvement) — populate it here only
from cases you actually observed comparing predictions to your ground
truth. Do not fabricate these; the brief weighs this section heavily
("worth more than the results table") specifically because it's the part
that's hardest to fake convincingly and easiest to catch if faked.

### Failure case 1
- **Video:**
- **Expected:**
- **Predicted:**
- **What went wrong:**
- **Root cause:**
- **Failed signal:**
- **Proposed improvement:**

### Failure case 2
- **Video:**
- **Expected:**
- **Predicted:**
- **What went wrong:**
- **Root cause:**
- **Failed signal:**
- **Proposed improvement:**

### Failure case 3
- **Video:**
- **Expected:**
- **Predicted:**
- **What went wrong:**
- **Root cause:**
- **Failed signal:**
- **Proposed improvement:**

## 7. Determinism

*(Report here after running each video twice: do the two runs produce the
same segment boundaries? Whisper's `vad_filter` and the fixed frame-sampling
grid are both deterministic given the same input file, so in principle
repeat runs on the same local file should match closely; VLM providers
introduce their own nondeterminism if temperature isn't pinned to 0 — note
what you observed, not what's merely expected.)*

## 8. What I cut and why

*(If you run out of the ~10 hour budget, record here what you deliberately
did not build and why — the brief explicitly rewards "what I cut and why"
over silently shipping something incomplete without saying so.)*
