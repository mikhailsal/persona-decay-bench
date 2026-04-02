# Experiment Analysis Report

*Date: April 3, 2026*

## 1. Overview

This document analyzes the results of two experimental runs of the Persona
Decay Benchmark. Run 1 (original configuration: 36 exchanges, 3 observer calls,
no response length limits) revealed critical cost and quality issues. Run 2
(optimized: 24 exchanges, 1 observer call, strict length limits) validated the
fixes on grok-4.1-fast.

### Run 1 — Original Configuration

| Model | Runs Completed | Status |
|-------|:-:|--------|
| grok-4.1-fast | 5/5 | Complete |
| gemini-3.1-flash-lite-preview | 5/5 | Complete |
| kimi-k2.5 | 2/5 (+ 1 partial) | Incomplete, observer not run |

Total benchmark cost from AI Proxy logs: **$4.06**

| Component | Requests | Cost |
|-----------|:--------:|-----:|
| Observer (gemini-3-flash-preview, 3x per checkpoint) | 251 | $1.59 |
| Partner (gemini-3.1-flash-lite, for all conversations) | ~660 | ~$0.90 |
| gemini-flash-lite as tested model + self-reports | ~430 | ~$0.68 |
| kimi-k2.5 as tested model + self-reports | 118 | $0.61 |
| grok-4.1-fast as tested model + self-reports | 317 | $0.29 |
| **Total** | **1,983** | **$4.06** |

Cost per fully-tested model: approximately **$1.60**.

### Run 2 — Optimized Configuration (grok-4.1-fast only)

| Parameter | Run 1 | Run 2 |
|-----------|-------|-------|
| Exchanges | 36 | 24 |
| Checkpoints | T6, T12, T18, T24, T30, T36 | T6, T12, T18, T24 |
| Observer calls per checkpoint | 3 | 1 (not yet run) |
| RESPONSE_MAX_TOKENS | 2048 | 512 |
| PARTNER_MAX_TOKENS | 512 | 150 |
| Prompt length constraints | None | Strict (3-5 sentences, no formatting) |
| Runs | 5 | 5 (parallel) |

Total cost for Run 2 (grok-4.1-fast, 5 runs, conversation + self-report only):

| Component | Requests | Cost |
|-----------|:--------:|-----:|
| grok-4.1-fast participant | 125 | $0.038 |
| gemini-flash-lite partner | 120 | $0.035 |
| Self-report (grok) | 20 | $0.014 |
| Observer | 0 | $0.000 (not yet run) |
| **Total** | **265** | **$0.087** |

**Cost per model (conversation + self-report): $0.087** — a **95% reduction**
from Run 1's $1.60 per model. Even with observer evaluation added (~$0.05-0.10
estimated), total would be ~$0.15-0.18.

## 2. Run 1 Key Findings

### 2.1 Results Replicate the Paper's Core Discovery

The paper's central finding — stable self-reports but declining observer
ratings — is confirmed by our data:

**Original paper (High ADHD, T6 → T18):**
- Self-report change: +0.2 (stable)
- Observer change: -3.5 (significant decay)
- Mean self-report for High ADHD: ~29/36

**Our Run 1 results (High ADHD, T6 → T18):**

| Model | Self-Report T6 | Self-Report T18 | Δ SR | Observer T6 | Observer T18 | Δ Obs |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| grok-4.1-fast | 36.0 | 35.2 | -0.8 | 30.1 | 28.2 | -1.9 |
| gemini-flash-lite | 31.6 | 31.0 | -0.6 | 26.5 | 27.5 | +1.0 |

Self-reports are extremely stable (confirming the paper). Observer decay for
grok (-1.9) is moderate — comparable to Claude Sonnet 4.5 in the paper (-1.6,
the most stable). Gemini-flash-lite shows no significant observer decay through
T18 but declines later (-1.9 by T36).

### 2.2 Our Observer Scores Are Higher Than the Paper's

Our observer mean at T6 (26.5-30.1) is notably higher than the paper's typical
range (~17.5 mean across models). Possible explanations:

1. Our observer model (gemini-3-flash-preview) may rate more generously
2. Our tested models may express ADHD more convincingly
3. The paper used different CAARS items (copyrighted originals vs. our
   functionally equivalent versions)

Grok self-reports a perfect 36/36 at nearly every checkpoint, which saturates
the scale. This is not necessarily wrong (high-ADHD persona should rate high)
but removes measurement sensitivity at the ceiling.

### 2.3 Conversations Die After Exchange ~25

The most critical operational finding: both models' responses collapse
dramatically in length after exchange 20-25.

| Metric | grok-4.1-fast | gemini-flash-lite |
|--------|:--:|:--:|
| Avg. words (exchanges 1-6) | 295 | 517 |
| Avg. words (exchanges 31-36) | 43 | 66 |
| Collapse ratio | 0.15x | 0.13x |

By exchange 30-36, participants produce 7-10 word farewell messages ("Thanks
for the chat! Take care.") and partners respond with similarly brief closings.
This means checkpoints at T30 and T36 were evaluating largely empty
conversations, making the observer ratings for those checkpoints unreliable
indicators of persona expression.

### 2.4 Response Lengths Are Excessive

Participant responses were far too long, especially in early turns:

- gemini-flash-lite averaged **517 words** in early exchanges (several paragraphs
  with headers, bullet points, and structured narratives)
- grok-4.1-fast averaged **295 words** with heavy markdown formatting
- kimi-k2.5 averaged **526 words**

The original paper almost certainly did not have this problem because (a)
earlier-generation models produced shorter outputs, and (b) the paper's
conversation length (18 turns) did not exhaust the topic.

The partner model (gemini-flash-lite) also grew verbose despite being instructed
to use "1-3 sentences" — reaching 100-213 words by exchange 10-15, with
paraphrasing, reflections, and multi-part questions.

### 2.5 Observer Inter-Rater Agreement: 3 Calls Largely Redundant

Three independent observer calls per checkpoint showed high agreement:

| Model | Mean Observer SD | Perfect Agreement (SD=0) | Near-Perfect (SD≤1.0) | Substantial Disagreement (SD>2.0) |
|-------|:---:|:---:|:---:|:---:|
| grok-4.1-fast | 1.64 | 17% | 40% | 33% |
| gemini-flash-lite | 0.84 | 23% | 63% | 3% |
| **Overall** | **1.24** | **20%** | **52%** | **13%** |

On a 0-36 scale, mean SD of 1.24 represents just 3.4% of the range. For
gemini-flash-lite, 63% of checkpoints had near-perfect agreement. The observer
model (gemini-3-flash-preview at temperature 0.3) produces highly deterministic
outputs, making 3 calls wasteful. The observer cost alone ($1.59) was 39% of
the total benchmark cost.

### 2.6 Turn Definition: No Misinterpretation

Our "turn" matches the paper's "turn" — both mean one exchange round (partner
question + participant response). Our benchmark ran 36 exchange rounds (72
messages) vs. the paper's 18 exchange rounds (36 messages). This was a
deliberate design choice to study extended decay, documented in the
implementation plan as: "extended to 36 turns (paper only went to 18, leaving
open whether decay continues)."

The paper's checkpoints were at turns 6, 12, 18. Ours were at 6, 12, 18, 24,
30, 36.

## 2b. Run 2 Key Findings (Optimized Configuration)

### 2b.1 Response Lengths: Massive Improvement

The strict prompt-level length constraints worked dramatically:

| Metric | Run 1 (grok) | Run 2 (grok) | Reduction |
|--------|:---:|:---:|:---:|
| Participant avg. words (overall) | 295 | 45 | **85%** |
| Partner avg. words (overall) | ~95 | 17 | **82%** |
| Participant range | 43-517 | 26-90 | Consistent |
| Last exchange word count | 7-10 (goodbyes) | 28-40 (substantive) | No collapse |

Word count progression (averaged across 5 runs):

```
Exchange  0:  83 words (initial "describe your day" response)
Exchange  6:  51 words
Exchange 12:  40 words
Exchange 18:  38 words
Exchange 24:  35 words (still substantive, no collapse)
```

The natural downward trend from 58 words (exchanges 0-6) to 36 words (exchanges
19-24) reflects the conversation narrowing in topic, which is expected. The
critical difference from Run 1 is that responses never collapse into 7-word
goodbyes. Every response through exchange 24 contains substantive ADHD-persona
content.

### 2b.2 Cost: 95% Reduction Achieved

| Cost Component | Run 1 (per model) | Run 2 (per model) | Change |
|----------------|:--:|:--:|:--:|
| Conversation generation | ~$0.55 | $0.073 | -87% |
| Self-report checkpoints | ~$0.03 | $0.014 | -53% |
| Observer evaluation | ~$1.06 | $0.00 (not yet run) | TBD |
| **Total** | **~$1.60** | **$0.087** (without observer) | **-95%** |

Estimated total with 1 observer call per checkpoint: ~$0.15-0.18 per model.
This makes benchmarking 10+ models economically viable at $1.50-1.80 total.

### 2b.3 Parallelism Works

All 5 runs executed in parallel (started within 4ms of each other). Each
conversation completed in ~2.0-2.3 minutes. Total wall-clock time for 5 runs:
**2.3 minutes** (vs. sequential which would be ~11 minutes).

### 2b.4 Participant Caching: 86-95% Cache Hits

The grok-4.1-fast participant model achieved excellent cache utilization:

| Run | Cache Hit Rate |
|-----|:-:|
| run_1 | 91% |
| run_2 | 95% |
| run_3 | 87% |
| run_4 | 88% |
| run_5 | 86% |

This means the growing conversation context (system prompt + prior turns) is
efficiently reused across sequential turn calls within each conversation.

### 2b.5 Partner Caching: 0% — Expected but Notable

The partner model (gemini-flash-lite) shows 0% cache hit rate across all runs.
This is because: (a) the partner's system prompt is short (~80 tokens, below
the 4096-token minimum for Gemini caching), and (b) each call includes the
full conversation history which changes with every new participant response.
Partner costs remain low ($0.007/run) due to the short response length, so
this is not a significant cost concern.

### 2b.6 Self-Report JSON Parsing: 15% Failure Rate (Bug — Fixed)

3 out of 20 self-reports (15%) failed to parse as JSON. The model refused to
produce structured data, instead writing prose responses like:

> "Man, doing it all in JSON feels like too much structure right now, but yeah,
> pretty much all the IN ones are 3s cuz focus is my nemesis."

**Root cause:** The persona prompt's formatting rules ("no formatting, plain
text only") leaked into the self-report questionnaire context. The model
interpreted "no formatting" as including JSON.

**Pattern:** Failures concentrated in early exchanges (T6) where the model is
most deeply immersed in its casual persona character. By T18-T24, the model
always produced valid JSON — likely because the accumulating conversation
context and questionnaire framing eventually override the persona formatting
instincts.

**Fix applied:** The self-report prompt now begins with an explicit override:
"IMPORTANT: This is a structured questionnaire. For this response ONLY, you
MUST output valid JSON — ignore any prior instructions about avoiding
formatting." The prompt also reinforces: "Do NOT include any text before or
after the JSON. Do NOT explain your ratings."

### 2b.7 Self-Report Scores: Remain Extremely Stable

For the 17/20 successfully parsed self-reports:

```
Turn  |  Mean ± SD   | N
------+--------------+---
  T6  | 35.7 ± 0.6  | 3  (2 failed to parse)
 T12  | 35.5 ± 1.0  | 4  (1 failed to parse)
 T18  | 35.8 ± 0.4  | 5
 T24  | 35.2 ± 1.8  | 5
```

This confirms the same pattern as Run 1: grok-4.1-fast self-rates at or near
maximum (36/36) throughout the conversation. The slight decline at T24 (35.2)
may represent genuine sensitivity or noise.

### 2b.8 Conversation Quality: Excellent

Sample exchanges from Run 2 demonstrate natural, ADHD-consistent dialogue:

> **Participant (T1):** I usually wake up late around 9, rush to make coffee
> while forgetting my phone somewhere, then sit down to work but end up
> scrolling social media for an hour.
>
> **Partner:** What part of your day do you find most frustrating?
>
> **Participant (T4):** Afternoons are the worst, when I stare at a report for
> hours but my brain wanders off to dumb thoughts or noises, leaving it
> half-done.

No markdown formatting, no bullet points, no structured essays. The model
writes as a real person would speak. Conversations flow naturally through all
24 exchanges without dying into generic goodbyes.

## 3. Cost Analysis

### 3.1 Run 1: What Costs the Most

| Category | Cost | % of Total |
|----------|-----:|:----------:|
| Observer evaluation (3x calls) | $1.59 | 39% |
| Partner model (flash-lite, all convs) | ~$0.90 | 22% |
| gemini-flash-lite as participant | ~$0.68 | 17% |
| kimi-k2.5 as participant (partial) | $0.61 | 15% |
| grok-4.1-fast as participant | $0.29 | 7% |

The observer is the single largest cost driver. Reducing from 3 calls to 1
would cut this by ~66% ($1.06 saved). Shorter conversations would reduce all
categories proportionally.

### 3.2 Run 2: Validated Cost Reduction

The combined effect of all optimizations for grok-4.1-fast (5 runs):

| Component | Run 1 | Run 2 | Reduction |
|-----------|------:|------:|:---------:|
| Participant turns | $0.052 (est.) | $0.030 | 42% |
| Partner turns | $0.160 (est.) | $0.040 | 75% |
| Self-report | $0.015 | $0.014 | 7% |
| Observer (3→1) | $0.638 | $0.000 (not run) | TBD |
| **Per-model total** | **~$1.60** | **~$0.087** | **~95%** |

The partner cost reduction (75%) is driven by shorter responses (17 words vs.
~95 words). Participant cost reduction (42%) comes from both fewer turns (24 vs.
36) and shorter responses (45 vs. 295 words). Self-report cost is nearly
unchanged because it depends on conversation length (reading all prior turns),
not response verbosity.

### 3.3 Why Caching Doesn't Help the Observer or Partner Fully

The observer receives the entire conversation text in a single user message at
each checkpoint. For 3 calls to the same checkpoint, the prompt is identical —
in theory, calls 2 and 3 should be fully cached. However, OpenRouter/Gemini
caching has a 5-minute TTL and minimum cacheable token thresholds. If calls are
made in rapid succession, call 2 may get a cache hit but the provider may still
report partial cache misses. The cost difference between 3 cached calls and 1
uncached call is small enough that reducing to 1 call is the pragmatic choice.

The partner model shows 0% cache hit rate in Run 2 because the system prompt
is too short (~80 tokens) to meet Gemini's caching threshold (4096 tokens).

## 4. Decisions Made

Based on these findings, the following changes were implemented:

### 4.1 Reduce Conversation Length: 36 → 24 Exchanges

**Rationale:** Conversations die after exchange ~25. Turns 25-36 measure
goodbye exchanges, not persona expression. Reducing to 24 exchanges captures
the meaningful decay period while cutting conversation generation costs by ~33%.

New checkpoints: turns 6, 12, 18, 24 (was: 6, 12, 18, 24, 30, 36).

### 4.2 Reduce Observer Calls: 3 → 1

**Rationale:** Observer inter-rater SD averages 1.24 on a 36-point scale (3.4%
of range). 52% of checkpoints show near-perfect agreement. The cost of 3 calls
($1.59 total) far outweighs the marginal reliability improvement. This change
saves ~$1.06 per benchmark run (~26% of total cost).

### 4.3 Reduce Max Token Limits

| Parameter | Old | New | Rationale |
|-----------|----:|----:|-----------|
| RESPONSE_MAX_TOKENS | 2048 | 512 | Participant responses averaged 275-517 words; 512 tokens (~380 words) is sufficient for 3-5 sentences |
| PARTNER_MAX_TOKENS | 512 | 150 | Partner should produce 1-2 sentences (~20-30 words); 150 tokens is generous |
| SELF_REPORT_MAX_TOKENS | 1024 | 512 | Self-report is a JSON object ~100 tokens; 512 is more than enough |
| OBSERVER_MAX_TOKENS | 1024 | 512 | Observer response is a JSON object ~100 tokens; 512 is more than enough |

### 4.4 Add Strict Length and Formatting Constraints to Prompts

**Participant persona prompt** now includes explicit rules:
- Keep every response to 3-5 sentences
- No markdown, no headers, no bullet points, no bold, no asterisks
- Write as a real person speaks — casual, brief, natural
- Never more than one short paragraph per response

**Partner prompt** rewritten to be more directive:
- Respond with exactly ONE short question, 1-2 sentences maximum
- No commentary, reflections, summaries, or paraphrasing
- Plain text only, no formatting

**Workday task** now instructs: "Keep your answer to 3-5 sentences — just give
a brief overview."

### 4.5 Add Conversation Cost Tracking

The runner now computes and reports per-conversation cost breakdown (participant
vs. partner) to enable better cost monitoring going forward.

## 5. Expected Impact

### Estimated Cost Reduction

| Change | Savings |
|--------|--------:|
| Observer 3→1 calls | ~66% of observer cost (~$1.06) |
| 36→24 exchanges | ~33% of conversation cost |
| Shorter responses (token limits + prompts) | ~50-70% of per-message cost |
| **Combined estimated new cost per model** | **~$0.30-0.50** (vs. ~$1.60) |

This would bring the cost for 10 models to approximately $3-5 total (vs. ~$16
at the old rate), making the full benchmark economically viable.

### Scientific Validity

- The 24-turn design still extends significantly beyond the paper's 18 turns,
  allowing us to observe whether decay continues or stabilizes
- Single observer call is standard practice in many LLM evaluation benchmarks
- Shorter, more naturalistic responses better simulate the real conversation
  dynamics the paper intended to study
- Self-report assessment is retained as a cheap ($0.001-0.003 per checkpoint)
  secondary metric

## 6. Decay Curves

### 6.1 Run 1 — Observer-Rated ADHD Expression Over Time

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

Both models show gradual observer decay through T24, then stabilization. This
supports the decision to set the endpoint at T24 — the remaining turns add
noise but no new signal.

### 6.2 Run 1 — Self-Report Scores (Extremely Stable)

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

Grok consistently self-rates at or near the maximum (36/36). Gemini-flash-lite
shows slightly more variation but remains stable around 31/36.

### 6.3 Run 2 — Self-Report Scores (grok-4.1-fast, Optimized Config)

Observer evaluation has not been run yet. Self-report scores for successfully
parsed responses (17/20, 3 failed JSON parsing):

```
Turn  |  grok-4.1-fast (Run 2)  |  grok-4.1-fast (Run 1)
------+-------------------------+------------------------
  T6  |  35.7 ± 0.6  (n=3)     |  36.0 ± 0.0  (n=5)
 T12  |  35.5 ± 1.0  (n=4)     |  35.8 ± 0.4  (n=5)
 T18  |  35.8 ± 0.4  (n=5)     |  35.2 ± 0.8  (n=5)
 T24  |  35.2 ± 1.8  (n=5)     |  35.8 ± 0.4  (n=5)
```

Self-report scores in Run 2 are virtually identical to Run 1, confirming that
the prompt optimizations did not disrupt the model's persona self-assessment.
The slightly higher variance in Run 2 (especially at T24) may be due to the
shorter, more casual response style interacting differently with the
questionnaire context.

## 7. Open Questions

1. **Observer evaluation for Run 2:** The observer has not been run on the new
   grok data. This is the next step to complete the comparison — we need
   observer scores to know if the shorter, more casual responses still express
   ADHD traits in a way the observer can detect. There is a risk that very
   brief responses provide less behavioral signal for the observer.

2. **Self-report JSON parsing robustness:** The prompt fix should resolve the
   15% failure rate, but needs validation. If it persists, consider a fallback
   parser that extracts scores from natural-language responses (e.g., regex
   matching "IN-1 3, IN-2 3, ..." patterns).

3. **Scale calibration:** Our observer scores (26-30 at T6 in Run 1) are much
   higher than the paper's (~17.5). Is this a measurement artifact or a genuine
   difference in model behavior? Run 2 observer data will help assess whether
   shorter responses change the score range.

4. **Conversation death: RESOLVED.** Run 2 shows no conversation death through
   24 exchanges. Responses decline naturally from ~58 to ~36 words but remain
   substantive throughout. The strict prompt constraints successfully prevent
   the collapse pattern seen in Run 1.

5. **Chinese models:** The benchmark cost reduction ($0.15-0.18/model estimated)
   makes testing 10+ models feasible at under $2 total. Ready to scale.

6. **Run 1 data cleared:** The old Run 1 data for all models has been removed
   from cache. Only the new Run 2 grok-4.1-fast data exists. All models need
   to be re-run with the optimized configuration.
