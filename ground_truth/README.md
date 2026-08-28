# Ground Truth — Instructions

This folder intentionally does **not** contain filled-in labels. The
assignment brief is explicit that hand-labeling the five test videos, and
defending how you resolved ambiguous cases, is part of what's being graded
(Section 4: *"How you resolve the ambiguous cases tells us more than your
F1 score does"*).

## What to do

1. Watch each of the five official test videos (URLs in the assignment
   brief, Section 4) yourself, start to finish.
2. Copy `ground_truth.template.json` to `ground_truth.json` in this folder.
3. For each video, add one entry per ad segment you observe, using the same
   field names as the pipeline's output contract (`start_s`, `end_s`,
   `ad_type`, `brand`, `description`). Timestamps in seconds from video start.
4. Use the **same `ad_type` taxonomy** you defined in `DESIGN.md` so your
   ground truth and your predictions can be compared apples-to-apples.
5. Where you hit one of the eight ambiguity-pack cases (Patreon mention,
   hoodie logo, short bumper, trailer-in-review, static sponsor read,
   whole-clip Reel ad, back-to-back sponsors, product placement), label it
   consistently with the ruling you wrote in `DESIGN.md` — and if labeling
   it forces you to refine that ruling, update `DESIGN.md` too. The two
   documents should agree.
6. Keep a note (in `EVAL.md`, not here) of anything genuinely ambiguous
   even after applying your own rulings — that note is more valuable to a
   reviewer than a confident-looking but arbitrary label.

## Video ID keys

Match these keys exactly so `scripts/run_evaluation.py` and your
`artifacts/<video_id>.json` prediction files line up:

| Key | Test set item |
|---|---|
| `video1_longform_youtube` | Long-form YouTube video |
| `video2_youtube_short` | YouTube Short |
| `video3_livestream_or_recording` | Live stream (or 20+ min recorded fallback — note which in EVAL.md) |
| `video4_instagram_reel_1` | Instagram Reel #1 |
| `video5_instagram_reel_2` | Instagram Reel #2 |

## What NOT to do

Do not ask an AI tool to watch the video for you and generate these labels.
The interview will ask you to defend specific boundary and taxonomy
decisions live — labels you can't explain will cost you more than labels
you never submitted.
