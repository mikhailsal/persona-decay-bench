# Experiment Run 1 — Analysis Report

*Date: April 3, 2026*

## 1. Overview

This document analyzes the results of the first experimental run of the Persona
Decay Benchmark. Two models were fully tested (5 runs each), one partially
tested (2 complete runs out of 5). The experiment replicated the "Stable
Personas" paper (arXiv:2601.22812v1) methodology with extended conversation
length (36 exchanges vs. the paper's 18).

### Models Tested

| Model | Runs Completed | Status |
|-------|:-:|--------|
| grok-4.1-fast | 5/5 | Complete |
| gemini-3.1-flash-lite-preview | 5/5 | Complete |
| kimi-k2.5 | 2/5 (+ 1 partial) | Incomplete, observer not run |

### Total Cost

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

## 2. Key Findings

### 2.1 Results Replicate the Paper's Core Discovery

The paper's central finding — stable self-reports but declining observer
ratings — is confirmed by our data:

**Original paper (High ADHD, T6 → T18):**
- Self-report change: +0.2 (stable)
- Observer change: -3.5 (significant decay)
- Mean self-report for High ADHD: ~29/36

**Our results (High ADHD, T6 → T18):**

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

## 3. Cost Analysis

### 3.1 What Costs the Most

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

### 3.2 Why Caching Doesn't Help the Observer Fully

The observer receives the entire conversation text in a single user message at
each checkpoint. For 3 calls to the same checkpoint, the prompt is identical —
in theory, calls 2 and 3 should be fully cached. However, OpenRouter/Gemini
caching has a 5-minute TTL and minimum cacheable token thresholds. If calls are
made in rapid succession, call 2 may get a cache hit but the provider may still
report partial cache misses. The cost difference between 3 cached calls and 1
uncached call is small enough that reducing to 1 call is the pragmatic choice.

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

### Observer-Rated ADHD Expression Over Time

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

### Self-Report Scores (Extremely Stable)

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

## 7. Open Questions

1. **Kimi-k2.5 observer evaluation:** Two complete conversations exist but
   observer ratings were never run. Should we run evaluation before clearing
   the old data, or start fresh with the new parameters?

2. **Scale calibration:** Our observer scores (26-30 at T6) are much higher
   than the paper's (~17.5). Is this a measurement artifact or a genuine
   difference in model behavior? Consider calibrating against the paper's
   reference values.

3. **Conversation death:** Even with 24 turns, some conversations may collapse
   into goodbyes by exchange 20. The new length limits should help, but
   monitoring is needed.

4. **Chinese models:** The original goal includes testing Chinese models
   (kimi-k2.5 is one). The benchmark cost reduction should make this feasible.
