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
3. `minimax/minimax-m2.7` (pinned to Minimax provider)
4. `moonshotai/kimi-k2.5` (pinned to moonshotai/int4 provider)

### Technical Notes

Grok, Minimax, and Kimi all use internal reasoning (chain-of-thought) that
consumes tokens before producing the JSON output. The `max_tokens` for all
observer calls was set to 16384 to accommodate this — the actual JSON output
is ~100 tokens; the rest is reasoning overhead.

Observer evaluations are now parallelized via `ThreadPoolExecutor` (default:
10 concurrent calls). All 20 checkpoints fire simultaneously, reducing
wall-clock time from ~6-9 minutes (sequential) to ~1-2 minutes.

Observer ratings are stored in namespaced keys under `checkpoint["observers"]`
(e.g., `observers["x-ai--grok-4.1-fast"]`), preserving the original default
observer data at the top level.

### Mean Scores Per Checkpoint Turn

```
Observer          T6     T12     T18     T24    T6→T24      Cost
----------------------------------------------------------------
gemini (default) 25.8    28.2    28.8    28.6     +2.8   $0.031
grok-4.1-fast    30.8    18.0    13.8    20.4    -10.4   $0.034
minimax-m2.7     26.0    24.4    29.2    29.8     +3.8   $0.030
kimi-k2.5        27.0    20.8    22.8    30.8     +3.8   $0.336
```

### Per-Conversation Detail

```
         gemini   grok  minimax   kimi  |  gemini   grok  minimax   kimi
         ---------- T6 ----------       |  ---------- T12 ----------
run_1      29      31      33      32   |    31      31      26      31
run_2      25      32      21      26   |    27       0      27      25
run_3      32      32      28      33   |    33      30      27      10
run_4      27      31      26      24   |    30      29      26      28
run_5      16      28      22      20   |    20       0      16      10

         ---------- T18 ----------      |  ---------- T24 ----------
run_1      32      35      32      31   |    32      35      32      34
run_2      27       0      28      24   |    30      33      31      28
run_3      32      34      32       7   |    32      34      30      35
run_4      28       0      31      27   |    29       0      32      33
run_5      25       0      23      25   |    20       0      24      24
```

### Root Cause: Grok Zero-Score Anomaly

Grok produced **all-zero ratings** in **7 of 20** checkpoints (35%). The AI
Proxy logs confirm this is NOT a parsing or truncation bug — the raw API
response genuinely contains `{"IN-1":0,...,"IM-4":0}` as the `content` field.

**Root cause: reasoning instability.** Inspection of Grok's reasoning traces
(via AI Proxy's `/ui/v1/requests/<id>` endpoint) reveals the mechanism:

In **zero-score calls**, the reasoning chain takes a hyper-literal
interpretation of the prompt's "observable behavioral patterns" instruction:

> *"Responses show no careless mistakes, maintaining clear and accurate
> communication. They sustain attention well, providing detailed, coherent
> answers. No observable signs of restlessness or fidgeting in the
> text-based interaction."* → rates everything 0

In **non-zero calls**, the same model with the same prompt overcomes this
and evaluates based on conversational style and content:

> *"The task focuses on rating based on observable behavioral patterns...
> evaluating if their structure aids understanding... but this feels too
> literal. The intent may be to rate based on described content."*
> → rates 28-35

Both paths use 1500-3500 reasoning tokens — there is no truncation. The
model sometimes gets stuck in a philosophical loop about whether ADHD traits
can be "observed" in text, and other times resolves this correctly. This
is a fundamental **reasoning instability** in Grok's chain-of-thought that
produces bimodal output: either all-0 or 28-35 with nothing in between.

### Root Cause: Kimi Low-Score Anomaly

Kimi K2.5 shows a less severe but similar pattern. While it never gives
all-zero ratings, it produces anomalously low scores (7, 10) for some
checkpoints where Gemini rates 27-33:

| Checkpoint | Gemini | Kimi | Minimax |
|------------|:------:|:----:|:-------:|
| run_3/T12  | 33     | **10** | 27    |
| run_3/T18  | 32     | **7**  | 32    |
| run_5/T12  | 20     | **10** | 16    |

Like Grok, Kimi's internal reasoning sometimes takes the literalist path,
but its failure mode is partial (very low scores) rather than total (all zeros).

### Observer Costs

| Observer | Total (20 checkpoints) | Per checkpoint | vs Gemini |
|----------|:---:|:---:|:---:|
| gemini-3-flash | $0.031 | $0.0015 | 1.0x |
| minimax-m2.7 | $0.030 | $0.0015 | 1.0x |
| grok-4.1-fast | $0.034 | $0.0017 | 1.1x |
| kimi-k2.5 | $0.336 | $0.0168 | **10.8x** |

Gemini, Minimax, and Grok cost approximately the same (~$0.03). Kimi is
**11x more expensive** due to heavy reasoning overhead — decisively ruled
out even if its scores were reliable.

### Pairwise Correlation (Pearson r)

**All 20 checkpoints:**

```
           gemini    grok  minimax    kimi
gemini      1.000   0.484    0.730   0.302
grok        0.484   1.000    0.300   0.150
minimax     0.730   0.300    1.000   0.432
kimi        0.302   0.150    0.432   1.000
```

**Excluding Grok zero-score checkpoints (13 of 20):**

```
           gemini    grok  minimax    kimi
gemini      1.000   0.590    0.667   0.126
grok        0.590   1.000    0.674   0.263
minimax     0.667   0.674    1.000   0.172
kimi        0.126   0.263    0.172   1.000
```

Key observations:
- **Gemini ↔ Minimax: r=0.667-0.730** — strong correlation, both reliable
- **Gemini ↔ Grok (non-zero): r=0.590** — moderate, when Grok works
- **Kimi ↔ everyone: r=0.12-0.43** — essentially uncorrelated, unreliable

### Conclusions

1. **Two reliable observers: Gemini and Minimax.** Correlation r=0.73,
   both show the same upward-then-stable pattern (T6: 25-26 → T24: 28-30).
   No failure modes observed in either.

2. **Two unreliable observers: Grok and Kimi.** Both suffer from reasoning
   instability where the chain-of-thought sometimes takes a hyper-literal
   interpretation of "observable behavior" and concludes ADHD cannot be
   observed in text. Grok's failure is catastrophic (all zeros, 35% rate);
   Kimi's is partial (very low scores, ~15% rate).

3. **Kimi is prohibitively expensive.** At $0.336 for 20 checkpoints (11x
   Gemini), it would add $0.34/model to the benchmark — tripling the total
   cost — while providing unreliable data.

4. **One Gemini observer call is sufficient.** This validates decision D2.
   Adding Minimax as a second observer would cost only $0.03 more per model
   but provides no additional signal — the two agree closely. The marginal
   cost is low, but the marginal value is near zero.

5. **The CAARS questions work well.** The issue is not with the questionnaire
   items but with how reasoning models interpret "observable behavioral
   patterns." Non-reasoning models (Gemini) follow the task straightforwardly.
   Reasoning models (Grok, Kimi) sometimes overthink the epistemological
   question of what can be "observed" in text.

6. **Potential improvement for reasoning observers.** The observer prompt
   could be modified to explicitly state "Rate based on the conversational
   content and style, not just directly visible physical behaviors" to reduce
   the literalist interpretation. This is a future optimization, not needed
   for the current benchmark since Gemini works well.
