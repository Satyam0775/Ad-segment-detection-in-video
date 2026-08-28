# DESIGN.md — Ad Segment Detector

> **How to use this document**: sections marked **[REVIEW — YOUR CALL]**
> contain a proposed default, not a final answer. The assignment brief is
> explicit that how you resolve the ambiguous cases matters more than your
> F1 score, and you will be asked to defend these choices live in the
> interview. Read each one, decide if you agree, and edit the text (not
> just the code) before submitting. Text elsewhere describes what was
> actually built and is accurate as written.

## 1. Definition of an ad

**Working definition used by this system**: a segment is an advertisement
if a third party (or the creator acting as a paid intermediary for a third
party) paid, or received consideration, for the audience's attention during
that segment — expressed through sponsor language, a call-to-action for a
paid product/service, a visual product placement, or a platform-inserted
slot the creator did not choose.

This deliberately **excludes** a creator promoting only themselves with no
third-party consideration (their own Patreon, their own merch, their own
other channel) — that is `self_promo`, a distinct taxonomy bucket, not "not
an ad." **[REVIEW — YOUR CALL]**: decide whether you want `self_promo` to
count toward your reported ad-detection metrics at all, or to be excluded
from scoring entirely. Either is defensible; be ready to say which and why.

## 2. Ad type taxonomy

Using the assignment's provided taxonomy, with the following operational
definitions (what triggers each label in this implementation):

| Type | Trigger in this implementation |
|---|---|
| `preroll` | Detected within the first ~2% of video duration |
| `midroll_sponsor_read` | Default for a fused candidate with ASR sponsor/CTA language mid-video |
| `product_placement` | Not auto-assigned by current rules — **[REVIEW]** see Section 4 below |
| `self_promo` | All contributing signal points are self-promo ASR phrases with no brand |
| `affiliate` | "affiliate" appears in transcript or OCR text near the candidate |
| `platform_inserted` | Not auto-assigned — see Section 4, item 5 |
| `bumper` | Short, isolated, scene-cut-anchored candidate below a confidence ceiling |
| `other` | Fallback, not currently reached by the rule set as written |

Two types (`product_placement`, `platform_inserted`) do not have an
automatic trigger in the current rule-based classifier — see the ambiguity
rulings below for why, and treat this as an honest gap: the classifier
defaults these to `midroll_sponsor_read` unless you extend
`src/detection/classifier.py` after deciding how you want to detect them.

## 3. Architecture

See `architecture/option_b_architecture.png`. Summary: acquisition (yt-dlp
+ ffmpeg) → parallel audio (Whisper ASR, timestamped) and frame sampling
(scene-cut-aware) → OCR + selective VLM on sampled frames → weighted signal
fusion → rule-based taxonomy classification → temporal segmentation (merge/
split) → validated JSON contract → viewer / evaluation.

**Signals chosen, and why:**
- **ASR is weighted highest** (`W_ASR_SPONSOR = 0.45` in `fusion.py`)
  because the brief itself notes spoken sponsor reads are the strongest
  signal in long-form video, and because a static-visual 90-second sponsor
  read (ambiguity item 6) has *no* other signal to rely on.
- **OCR** catches on-screen promo text (discount codes, "sponsored" banners)
  that ASR misses when the host doesn't say it aloud, and that a VLM call
  would be needlessly expensive to catch instead.
- **VLM** is the most expensive and least-called signal by design
  (`VLM_MAX_CALLS_PER_VIDEO`, selective calling in `analyzer.py`) — it fires
  on ASR/OCR-flagged candidates plus a light background sample, not on
  every frame, specifically to keep the $10 test-set budget honest.
- **Scene cuts** are cheap (`ffmpeg` `select=gt(scene,...)`, one decode
  pass) and used two ways: (a) to densify frame sampling right around a cut
  so short bumpers aren't skipped between two uniform samples, and (b) as a
  small confidence nudge in fusion — never enough alone to trigger a
  detection.

**What I tried that did not work**: *(fill in from your own runs — do not
invent this. If you tried a different scene-cut threshold, a different ASR
model size, or a different fusion weight and it made things worse, that
belongs here. This section is explicitly graded and an honest "nothing
yet, I ran out of time" is better than a fabricated experiment.)*

## 4. The ambiguity pack — rulings **[REVIEW — YOUR CALL on every item]**

1. **"If you like this channel, join my Patreon."**
   *Proposed ruling*: not a third-party ad — `self_promo`. No money changes
   hands from an advertiser; it's the creator asking for direct support.
   *Signal*: ASR self-promo phrase match, no brand guess, no sponsor/CTA
   phrase co-occurring.

2. **Host wears a hoodie with their own merch logo for the whole video.**
   *Proposed ruling*: not an ad segment at all — it's not a *segment*, it's
   ambient branding with no start/end and no distinct call to action. We
   do not emit a segment for this. If the host explicitly calls out the
   merch with a CTA ("link in bio for this hoodie"), *that specific moment*
   becomes a `self_promo` segment via the self-promo ASR phrase rule.

3. **A 1.4-second animated "sponsored by" bumper card.**
   *Proposed ruling*: counts as an ad (`bumper` type), and this is exactly
   why frame sampling is not naive uniform sampling — see Section 3 and
   `src/media/frames.py`. `BUMPER_GUARD_WINDOW_S` forces dense sampling
   around every detected scene cut so a sub-2-second card can't fall
   entirely between two uniform samples. **This is one of the two items the
   brief says separates submissions — be ready to explain the exact
   sampling math in the interview, not just wave at the setting.**

4. **A movie review that plays 40 seconds of the official trailer.**
   *Proposed ruling*: this is genuinely ambiguous and depends on framing —
   is the trailer the *content being reviewed* (not an ad) or *promotional
   material for the studio* (is an ad, `platform_inserted`/`other`)? Default
   behavior: not auto-flagged (no current signal distinguishes "trailer
   clip in a review" from any other B-roll). If your own labeling judges
   this should be flagged, you'll need to extend the OCR/VLM prompt to look
   for studio logos/trailer-card framing — decide and document your call.

5. **A 90-second sponsor read where the visual never changes.**
   *Proposed ruling*: still an ad — this is why ASR carries the highest
   fusion weight (Section 3). No visual signal is expected or required;
   the segment is detected and bounded entirely by transcript timing.
   **This is the second of the two brief-flagged separating items.**

6. **A pre-roll ad served by the platform, never in the downloaded stream.**
   *Proposed ruling*: undetectable and out of scope — we only see what
   yt-dlp downloads. Document this limitation in EVAL.md rather than
   claiming coverage we don't have.

7. **A Reel that is an ad from frame one to the last frame.**
   *Proposed ruling*: emit one segment, `start_s=0.0`,
   `end_s=duration_s` (see `segmenter.py` docstring). Confirm this matches
   how you hand-labeled the two Reel test cases.

8. **Two sponsors back-to-back with no gap.**
   *Proposed ruling*: two segments if the brand/context clearly changes
   between them even within a small gap (`BACK_TO_BACK_SPLIT_GAP_S` in
   `fusion.py`/`segmenter.py`); one segment if the gap is small and nothing
   distinguishes them (can't tell they're different sponsors). This is a
   real, imperfect heuristic — say so plainly in the interview rather than
   defending it as more principled than it is.

## 5. Frame sampling

Base interval `FRAME_SAMPLE_INTERVAL` (default 2.0s) everywhere; short-form
video (< `SHORT_VIDEO_DURATION_THRESHOLD_S`, default 120s) uses the dense
interval throughout instead, since Shorts/Reels are short enough that dense
sampling is cheap and the ad content is often the entire clip. Around every
detected scene cut, sampling densifies to `FRAME_DENSE_SAMPLE_INTERVAL`
(default 0.5s) for a `BUMPER_GUARD_WINDOW_S` (default 3s) window — this is
the concrete mechanism for not missing short bumpers (ambiguity item 3).

**What we still miss**: a bumper that appears with *no* scene cut (a slow
cross-fade into and out of the card) would not get the dense-sampling
boost. This is a known gap, not a claimed strength.

## 6. Signal fusion — the math

See `src/detection/fusion.py`. Fused confidence per timestamp is a weighted
sum: `0.45×ASR_sponsor + 0.30×ASR_cta + 0.20×ASR_self_promo + 0.25×OCR +
0.35×VLM + 0.05×scene_cut_bonus`, clipped to [0,1]. A segment is only
emitted if the fused confidence clears `DETECTION_CONFIDENCE_THRESHOLD`
(default 0.55) — so no single signal alone (max weight 0.45) can trigger a
detection; agreement across signals is required by construction, except for
CTA+scene-cut combinations which can reach 0.35, still below threshold.
This was a deliberate choice to bias toward precision over recall given the
brief's framing that an honest 70%-recall system with clear failure
analysis beats an unverified 95% claim.

## 7. Temporal segmentation

Points above threshold are merged if the gap between consecutive points is
≤ `SEGMENT_MERGE_GAP_S` (default 3s), with the back-to-back-different-brand
exception in item 8 above. See `src/detection/segmenter.py::cluster_points`
for the exact algorithm; it's a straightforward single-pass greedy merge,
not a global optimization — flagged here so it isn't mistaken for something
more sophisticated than it is.

## 8. Cost, latency, and scaling

*(Fill in real numbers here once you've run the five test videos — do not
estimate or invent these. The mock VLM provider costs $0 by construction;
Gemini/OpenRouter costs are placeholder per-call estimates in
`gemini_provider.py`/`openrouter_provider.py` until you've measured actual
spend and updated `cost_per_call_usd`.)*

**What would need to change at ~1,000 videos/day**:
- Whisper on CPU (`WHISPER_DEVICE=cpu`) is almost certainly the bottleneck;
  a GPU worker pool or a hosted ASR API would be needed.
- The pipeline currently runs fully synchronously per request inside the
  FastAPI handler — at volume this needs a queue and worker processes
  (explicitly out of scope for this submission per the "no Celery/Redis"
  constraint, but worth naming as the next real step).
- VLM calls would need batching or a cheaper default model, since even at
  a capped `VLM_MAX_CALLS_PER_VIDEO=40`, 1000 videos/day is 40,000 calls/day.

## 9. Failure modes

See `EVAL.md` for the three required, observed failure cases — not invented
here.

## 10. What I'd build next with two more weeks

- A real evaluation of the `product_placement` and `platform_inserted`
  taxonomy gaps noted in Section 2, likely via a more targeted VLM prompt.
- Tier 2 live-stream windowing tested against an actually-live URL (only
  designed, not validated live, in this submission — see README limitations).
- Replace the greedy segment merge/split heuristic with something that
  scores candidate groupings against multiple hypotheses, rather than a
  single left-to-right pass.
- A small labeled dev set beyond the five official videos to tune fusion
  weights properly instead of by inspection.

## 11. AI tool use disclosure

Per assignment Section 9: the initial scaffolding, boilerplate (FastAPI
routing, Pydantic schemas, provider abstractions, test structure) was
generated with AI assistance (Claude) from a detailed spec. The core
judgment calls — the ambiguity-pack rulings, fusion weight rationale,
sampling strategy, and taxonomy decisions above — are represented here as
my own reasoning and are explicitly marked `[REVIEW — YOUR CALL]` where
they still need my confirmation before submission. Ground truth labels were
not AI-generated; I labeled the five test videos myself (see
`ground_truth/README.md`). **[REVIEW]**: rewrite this paragraph in your own
voice before submitting, and make sure it's factually true of what you
actually did.
