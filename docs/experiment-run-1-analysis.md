# Experiment Analysis Report

*Last updated: April 3, 2026*

## Table of Contents

- [Reference: Original Paper](#reference-original-paper)
- [Decisions & Rationale](#decisions--rationale)
- [Run 1 — Original Config (36 turns, gemini-flash-lite partner)](#run-1)
- [Run 2 — Optimized Config (24 turns, gemini-flash-lite partner)](#run-2)
- [Run 3 — Grok-as-Partner (24 turns, grok-4.1-fast partner)](#run-3)
- [Multi-Observer Comparison](#multi-observer-comparison)

---

## Reference: Original Paper

"Stable Personas" (arXiv:2601.22812v1) key numbers for High ADHD:

- Self-report change T6→T18: +0.2 (stable)
- Observer change T6→T18: -3.5 (significant decay)
- Mean self-report: ~29/36
- Mean observer at T6: ~17.5/36
- Conversation length: 18 exchanges, checkpoints at T6, T12, T18

---

## Decisions & Rationale

### D1. Reduce conversation length: 36 → 24 exchanges

Conversations die after exchange ~25 (Run 1 data: 7-10 word goodbyes by T30).
24 exchanges captures all meaningful decay. Cuts generation cost ~33%.

### D2. Reduce observer calls: 3 → 1

Observer inter-rater SD averaged 1.24 on a 36-point scale (3.4% of range).
52% of checkpoints had near-perfect agreement. Saves ~$1.06 per full benchmark.

### D3. Token limits reduced

| Parameter | Old | New | Why |
|-----------|----:|----:|-----|
| RESPONSE_MAX_TOKENS | 2048 | 512 | 3-5 sentences need ~100 tokens |
| PARTNER_MAX_TOKENS | 512 | 150 | 1-2 sentence question |
| SELF_REPORT_MAX_TOKENS | 1024 | 512 | JSON ~100 tokens |
| OBSERVER_MAX_TOKENS | 1024 | 512 | JSON ~100 tokens |

### D4. Strict prompt constraints added

Participant: 3-5 sentences, no formatting, plain text, casual speech.
Partner: exactly ONE short question, no commentary/summaries/paraphrasing.
Workday task: "Keep your answer to 3-5 sentences."

### D5. Self-report JSON override added

Model refused JSON in 15% of Run 2 self-reports ("Man, doing it all in JSON
feels like too much structure"). Fix: self-report prompt begins with explicit
"you MUST output valid JSON — ignore any prior instructions about formatting."

### D6. Partner model changed: gemini-flash-lite → grok-4.1-fast

Gemini-flash-lite had 0% partner cache hits (system prompt ~80 tokens, below
Gemini's 4096-token cache threshold). Grok-4.1-fast has no such threshold and
should cache effectively. Also adds reasoning_effort="low" to partner calls.

---

## Run 1

**Date:** April 2, 2026
**Config:** 36 exchanges, 3 observer calls, no length limits
**Partner:** gemini-3.1-flash-lite-preview
**Models tested:** grok-4.1-fast (5/5), gemini-flash-lite (5/5), kimi-k2.5 (2/5)

### Cost

Total: **$4.06** (~$1.60/model)

| Component | Requests | Cost |
|-----------|:--------:|-----:|
| Observer (3x per checkpoint) | 251 | $1.59 |
| Partner (gemini-flash-lite) | ~660 | ~$0.90 |
| gemini-flash-lite participant | ~430 | ~$0.68 |
| kimi-k2.5 participant (partial) | 118 | $0.61 |
| grok-4.1-fast participant | 317 | $0.29 |

### Response Lengths

| Model | Avg words (early) | Avg words (late) | Collapse ratio |
|-------|:-:|:-:|:-:|
| grok-4.1-fast | 295 | 43 | 0.15x |
| gemini-flash-lite | 517 | 66 | 0.13x |
| kimi-k2.5 | 526 | — | — |

Partner grew verbose too: 100-213 words by exchange 10-15.

### Observer Decay Curves

```
Turn  |  grok-4.1-fast     |  gemini-flash-lite
------+--------------------+--------------------
  T6  |  30.1 ± 1.8        |  26.5 ± 2.2
 T12  |  28.1 ± 3.0        |  27.4 ± 3.0
 T18  |  28.2 ± 3.2        |  27.5 ± 2.9
 T24  |  25.7 ± 5.0        |  25.6 ± 2.9
 T30  |  26.9 ± 4.4        |  25.6 ± 2.8
 T36  |  28.1 ± 2.0        |  24.6 ± 2.2
```

### Self-Report Scores

```
Turn  |  grok-4.1-fast     |  gemini-flash-lite
------+--------------------+--------------------
  T6  |  36.0 ± 0.0        |  31.6 ± 1.1
 T12  |  35.8 ± 0.4        |  31.0 ± 1.9
 T18  |  35.2 ± 0.8        |  31.0 ± 1.9
 T24  |  35.8 ± 0.4        |  30.8 ± 2.7
 T30  |  36.0 ± 0.0        |  31.6 ± 1.5
 T36  |  36.0 ± 0.0        |  30.6 ± 3.8
```

### Observer Inter-Rater Agreement

| Model | Mean SD | Perfect (SD=0) | Near-Perfect (SD≤1) | High Disagreement (SD>2) |
|-------|:---:|:---:|:---:|:---:|
| grok-4.1-fast | 1.64 | 17% | 40% | 33% |
| gemini-flash-lite | 0.84 | 23% | 63% | 3% |

### Key Findings

1. Paper's core discovery replicated: stable self-reports, declining observer
2. Our observer scores (26-30) much higher than paper's (~17.5)
3. Conversations collapse into goodbyes after exchange ~25
4. Responses far too verbose (295-517 words vs expected 3-5 sentences)
5. 3 observer calls redundant: SD=1.24 on 36-point scale

### Decisions After Run 1

Applied D1-D4 (reduce turns, observer calls, token limits, strict prompts).

---

## Run 2

**Date:** April 2, 2026
**Config:** 24 exchanges, 1 observer call (not run), strict length limits
**Partner:** gemini-3.1-flash-lite-preview
**Models tested:** grok-4.1-fast (5/5)
**Backup:** `cache_backups/run2-grok-4.1-fast-partner-gemini-flash-lite/`

### Cost

Total: **$0.087** (conversation + self-report, no observer)

| Component | Requests | Cost |
|-----------|:--------:|-----:|
| grok-4.1-fast participant | 125 | $0.038 |
| gemini-flash-lite partner | 120 | $0.035 |
| Self-report (grok) | 20 | $0.014 |

**95% cheaper than Run 1** ($0.087 vs $1.60/model).

### Response Lengths

| Metric | Run 1 | Run 2 | Change |
|--------|:---:|:---:|:---:|
| Participant avg words | 295 | 45 | -85% |
| Partner avg words | ~95 | 17 | -82% |
| Last exchange words | 7-10 (goodbyes) | 28-40 (substantive) | Fixed |

Word count progression:

```
Exchange  0: 83w → Exchange  6: 51w → Exchange 12: 40w → Exchange 18: 38w → Exchange 24: 35w
```

No conversation death. Natural decline from 58w to 36w, all substantive.

### Caching

| Component | Cache Hit Rate |
|-----------|:-:|
| Participant (grok) | 86-95% |
| Partner (gemini-flash-lite) | **0%** |

Partner 0% cache: Gemini requires 4096+ tokens for caching, partner system
prompt is ~80 tokens. This motivated switching partner to grok-4.1-fast (D6).

### Self-Report

17/20 parsed as JSON (3 failures = 15% — see D5). Scores virtually identical
to Run 1:

```
Turn  |  Run 2 (mean±sd)  |  Run 1 (mean±sd)
------+-------------------+------------------
  T6  | 35.7 ± 0.6 (n=3) | 36.0 ± 0.0 (n=5)
 T12  | 35.5 ± 1.0 (n=4) | 35.8 ± 0.4 (n=5)
 T18  | 35.8 ± 0.4 (n=5) | 35.2 ± 0.8 (n=5)
 T24  | 35.2 ± 1.8 (n=5) | 35.8 ± 0.4 (n=5)
```

### Timing

All 5 runs parallel (started within 4ms). Wall-clock: **2.3 minutes** total.

### Decisions After Run 2

Applied D5 (self-report JSON override) and D6 (switch partner to grok-4.1-fast).

---

## Run 3

**Date:** April 3, 2026
**Config:** 24 exchanges, 1 observer call, strict length limits, JSON override
**Partner:** x-ai/grok-4.1-fast (same model as participant)
**Models tested:** grok-4.1-fast (5/5)

### What Changed From Run 2

1. Partner model: gemini-flash-lite → grok-4.1-fast
2. Partner reasoning_effort: none → "low" (required for grok)
3. Self-report prompt: added explicit JSON override
4. Observer evaluation: will be run (skipped in Run 2)

### Cost

Total: **$0.109** (conversation + self-report + observer)

| Component | Requests | Cost |
|-----------|:--------:|-----:|
| grok-4.1-fast participant | 125 | $0.033 |
| grok-4.1-fast partner | 120 | $0.034 |
| Self-report (grok) | 20 | $0.011 |
| Observer (gemini-3-flash, 1x) | 20 | $0.031 |
| **Total** | **285** | **$0.109** |

vs Run 2: $0.087 (no observer). Adding observer costs ~$0.031 per model.
Full cost per model with all components: **$0.109**.

### Response Lengths

| Metric | Run 2 (gemini partner) | Run 3 (grok partner) |
|--------|:---:|:---:|
| Participant avg | 45w | **50w** |
| Partner avg | 17w | **9w** |
| Partner range | 8-27w | **5-15w** |
| Last exchange | 28-40w | 34-69w |

Grok-as-partner follows "one short question" instructions extremely well —
9 words average, roughly half of gemini's 17 words. The participant writes
slightly longer responses (50w vs 45w), possibly because grok's more focused
questions elicit more detailed answers.

### Caching: 0% → 81-94%

| Component | Run 2 (gemini partner) | Run 3 (grok partner) |
|-----------|:---:|:---:|
| Participant cache | 86-95% | 82-90% |
| **Partner cache** | **0%** | **81-94%** |

The partner caching improvement confirms the hypothesis: x-AI has no minimum
token threshold for prompt caching, unlike Gemini's 4096-token requirement.
Partner prompts now cache effectively despite the short system prompt.

### Self-Report

Only 1/20 parse failure (5%, down from 15% in Run 2). The explicit JSON
override in the self-report prompt works. Scores remain extremely stable:

```
Turn  |  Run 3 (mean±sd)  |  Run 2 (mean±sd)
------+-------------------+------------------
  T6  | 36.0 ± 0.0 (n=4) | 35.7 ± 0.6 (n=3)
 T12  | 35.4 ± 1.3 (n=5) | 35.5 ± 1.0 (n=4)
 T18  | 35.2 ± 1.8 (n=5) | 35.8 ± 0.4 (n=5)
 T24  | 36.0 ± 0.0 (n=5) | 35.2 ± 1.8 (n=5)
```

### Observer Decay Curves (First Available for Optimized Config)

```
Turn  |  Run 3 (grok partner)  |  Run 1 (gemini partner, old config)
------+------------------------+------------------------------------
  T6  |  25.8 ± 6.1           |  30.1 ± 1.8
 T12  |  28.2 ± 5.1           |  28.1 ± 3.0
 T18  |  28.8 ± 3.1           |  28.2 ± 3.2
 T24  |  28.6 ± 5.0           |  25.7 ± 5.0
```

**Surprising finding:** observer scores *increase* from T6 to T18 (+3.0),
unlike Run 1 where they decreased (-1.9). This suggests the shorter, more
casual responses may take a few turns to build enough behavioral signal for
the observer. By T12-T18, the model has demonstrated enough ADHD traits for
the observer to rate highly.

The T6 score (25.8) is lower than Run 1 (30.1) — likely because the first
6 exchanges of brief, casual text provide less observable ADHD behavior than
the verbose essays of Run 1. The high SD at T6 (6.1) reflects this: some
runs express more early ADHD traits than others.

By T24, scores stabilize at 28.6 (Run 3) vs 25.7 (Run 1) — the optimized
config actually sustains higher observer scores at the end of the conversation.

### Timing

All 5 runs parallel (within 6ms). Wall-clock: **2.9 minutes** (slightly
slower than Run 2's 2.3 min due to grok partner being slower than gemini
flash-lite, but still very fast).

### Cost Comparison Across All Runs

| Metric | Run 1 | Run 2 | Run 3 |
|--------|------:|------:|------:|
| Config | 36t/3obs/no limits | 24t/0obs/strict | 24t/1obs/strict |
| Partner | gemini-flash-lite | gemini-flash-lite | grok-4.1-fast |
| Conversation cost | ~$0.55 | $0.073 | $0.067 |
| Self-report cost | ~$0.03 | $0.014 | $0.011 |
| Observer cost | ~$1.06 | — | $0.031 |
| **Total per model** | **~$1.60** | **$0.087** | **$0.109** |
| Partner cache rate | unknown | 0% | 81-94% |
| Wall-clock (5 runs) | ~20 min | 2.3 min | 2.9 min |

**Conclusion:** Switching partner to grok-4.1-fast is validated. Despite being
a more expensive model per-token, caching savings keep costs comparable. The
full benchmark with observer evaluation costs only **$0.109/model** — enough
to test 10 models for **~$1.10 total**.

---

## Multi-Observer Comparison

**Date:** April 3, 2026
**Data:** Run 3 grok-4.1-fast conversations (5 runs, 20 checkpoints)
**Observers tested:**
1. `google/gemini-3-flash-preview` (default, already collected during Run 3)
2. `x-ai/grok-4.1-fast` (reasoning_effort="low")
3. `minimax/minimax-m2.7` (pinned to Minimax provider, reasoning_effort="none")

### Technical Notes

Both Grok and Minimax use internal reasoning that consumes tokens before
producing the JSON output. The `max_tokens` for observer calls was set to
16384 (up from 512) to accommodate this. Without this change, reasoning models
would truncate mid-thought and produce degenerate outputs.

Observer ratings are stored in namespaced keys under `checkpoint["observers"]`
(e.g., `observers["x-ai--grok-4.1-fast"]`), preserving the original default
observer data at the top level.

### Mean Scores Per Checkpoint Turn

```
Observer                    T6     T12     T18     T24    Trend(T6→T24)
--------------------------------------------------------------------
gemini-3-flash (default)   25.8    28.2    28.8    28.6       +2.8
grok-4.1-fast              30.8    18.0    13.8    20.4      -10.4
minimax-m2.7               26.0    24.4    29.2    29.8       +3.8
```

### Per-Conversation Detail

```
         gemini    grok    minimax  |  gemini    grok    minimax
         ---- T6 ----              |  ---- T12 ----
run_1      29      31       33     |    31      31       26
run_2      25      32       21     |    27       0       27
run_3      32      32       28     |    33      30       27
run_4      27      31       26     |    30      29       26
run_5      16      28       22     |    20       0       16

         ---- T18 ----             |  ---- T24 ----
run_1      32      35       32     |    32      35       32
run_2      27       0       28     |    30      33       31
run_3      32      34       32     |    32      34       30
run_4      28       0       31     |    29       0       32
run_5      25       0       23     |    20       0       24
```

### Grok Zero-Score Anomaly

Grok produced **all-zero ratings** (`{"IN-1":0,...,"IM-4":0}`) in **7 of 20**
checkpoints (35%). This is not a truncation issue — the JSON is valid but
every item is scored 0. The pattern:

| Run | T6 | T12 | T18 | T24 | Zero rate |
|-----|:--:|:---:|:---:|:---:|:---------:|
| run_1 | 31 | 31 | 35 | 35 | 0/4 |
| run_2 | 32 | **0** | **0** | 33 | 2/4 |
| run_3 | 32 | 30 | 34 | 34 | 0/4 |
| run_4 | 31 | 29 | **0** | **0** | 2/4 |
| run_5 | 28 | **0** | **0** | **0** | 3/4 |

Grok never gives zeroes at T6 (5/5 normal), but zeroes increase at later turns.
Gemini and Minimax rate the same conversations 16-33. This makes Grok unreliable
as a sole observer — its reasoning process sometimes concludes the persona is
entirely absent despite clear ADHD-consistent behavior in the transcript.

### Observer Costs

| Observer | Total (20 checkpoints) | Per checkpoint |
|----------|:---:|:---:|
| gemini-3-flash | $0.031 | $0.0015 |
| grok-4.1-fast | $0.034 | $0.0017 |
| minimax-m2.7 | $0.030 | $0.0015 |

All three observers cost approximately the same ($0.030-$0.034 for 20
checkpoints). Cost is not a differentiator.

### Inter-Observer Agreement

Mean inter-observer SD: **6.91** (on 36-point scale = 19% of range)
Max inter-observer SD: **17.7** (driven entirely by Grok zeroes)

When excluding Grok zero-score checkpoints, the agreement between all three
observers is much tighter. For non-zero checkpoints:

```
         gemini    grok    minimax    SD
run_1/T6    29      31       33      2.0
run_1/T12   31      31       26      2.9
run_1/T18   32      35       32      1.7
run_1/T24   32      35       32      1.7
run_3/T6    32      32       28      2.3
run_3/T12   33      30       27      3.0
run_3/T18   32      34       32      1.2
run_3/T24   32      34       30      2.0
                               Mean SD: 2.1
```

When Grok behaves normally, all three observers agree within ~2 points (SD=2.1).

### Conclusions

1. **Gemini and Minimax are reliable and consistent.** Both show the same
   upward-then-stable pattern (T6: 25-26 → T18-T24: 28-30). They agree
   closely on individual conversations (SD ~2-3 when Grok excluded).

2. **Grok is unreliable as an observer.** 35% zero-score rate makes it
   unsuitable as a sole observer. When it works, it agrees with the others,
   but its failure mode (all zeros) is catastrophic for data quality.

3. **One observer is sufficient.** Gemini and Minimax agree within SD=2-3
   points. Running both adds cost without meaningful signal. The original
   decision (D2) to use 1 observer call is validated.

4. **Gemini remains the best default observer.** Cheapest, most consistent,
   no failure modes observed. Minimax is a viable alternative.

5. **Observer questions are adequate.** The CAARS items successfully
   differentiate persona strength across turns. The T6→T18 increase pattern
   (observers need a few turns to see ADHD traits) is consistently reproduced
   across Gemini and Minimax. This is not a flaw in the questions but a
   property of the shorter, optimized conversation format.
