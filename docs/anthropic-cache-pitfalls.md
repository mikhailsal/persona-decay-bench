# Anthropic Prompt Caching Pitfalls

> Investigation conducted April 2, 2026 during persona-decay-bench development.
> Model tested: `anthropic/claude-haiku-4.5` via OpenRouter with AI Proxy v2.

## TL;DR

Anthropic prompt caching can silently fail when request parameters indirectly
toggle extended thinking on or off.  **Changing `temperature` or `max_tokens`
between requests that share the same message prefix can cause 100% cache
misses** — not because those parameters are in the cache key, but because they
cause Anthropic to enable or disable thinking, and thinking parameter changes
invalidate message-level cache entries.

---

## The Problem We Observed

Self-report checkpoint requests shared an identical message prefix with the
preceding conversation turn (byte-for-byte verified via SHA-256 hashes on each
message block), yet always resulted in a full cache re-write instead of a cache
read:

| Request         | Messages | Input Tokens | Cached | Cache Write | Cost     |
|-----------------|----------|-------------|--------|-------------|----------|
| Turn 6 (conv)   | 14       | 7,395       | 6,487 (88%) | 898     | $0.0062  |
| Self-report      | 16       | 10,297      | 0 (0%)      | 10,294  | $0.0136  |

The only differences between the two requests were:
- `temperature`: 1.0 (conv) vs 0.3 (self-report)
- `max_tokens`: 2048 (conv) vs 1024 (self-report)

Both sent `reasoning: {"effort": "low"}` and identical `cache_control`.

---

## Root Cause: Indirect Thinking Parameter Toggle

### Chain of Events

1. **OpenRouter translates `reasoning.effort` into Anthropic's `budget_tokens`**
   using the formula:

   ```
   budget_tokens = max(min(max_tokens × effort_ratio, 128000), 1024)
   ```

   where `effort_ratio` for `"low"` is **0.2**.

   Source: [OpenRouter Reasoning Tokens docs](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

2. **With `max_tokens=2048`** (conversation turns):
   - `budget_tokens = max(min(2048 × 0.2, 128000), 1024) = max(409, 1024) = 1024`
   - `max_tokens (2048) > budget_tokens (1024)` — **thinking is enabled**

3. **With `max_tokens=1024`** (self-report):
   - `budget_tokens = max(min(1024 × 0.2, 128000), 1024) = max(204, 1024) = 1024`
   - `max_tokens (1024) == budget_tokens (1024)` — **NOT strictly greater**
   - OpenRouter docs: "max_tokens must be strictly higher than the reasoning
     budget to ensure there are tokens available for the final response"
   - Result: **thinking is silently disabled**

4. **`temperature=0.3`** delivers the final blow:
   - Anthropic docs: "Thinking isn't compatible with `temperature` or `top_k`
     modifications"
   - Anthropic API error (when called directly): "temperature may only be set
     to 1 when thinking is enabled"
   - OpenRouter silently handles this by disabling thinking rather than
     returning an error

5. **Thinking parameter changes invalidate message cache**:
   - Anthropic docs: "Changes to thinking parameters (enabled/disabled or
     budget allocation) invalidate message cache breakpoints"
   - The conversation turn had thinking enabled; the self-report had thinking
     disabled → cache invalidated

### Empirical Proof

We ran controlled experiments with identical message prefixes:

| Run | temperature | max_tokens | Reasoning Tokens | Cache Hit |
|-----|-------------|------------|-----------------|-----------|
| 1   | 0.3         | 1024       | 0               | MISS      |
| 2   | 1.0         | 1024       | 0               | MISS      |
| 3   | 1.0         | 2048       | 670             | 81% HIT   |

Key observations:
- Run 1: Both temperature AND max_tokens incompatible → thinking disabled → 0 reasoning tokens → cache miss
- Run 2: temperature OK, but `max_tokens == budget_tokens` → thinking silently disabled → 0 reasoning tokens → cache miss
- Run 3: Both parameters compatible → thinking enabled → 670 reasoning tokens → cache hit

---

## The Fix

For any request that should reuse a cache entry written by a prior request:

1. **Match `temperature` exactly** — must be 1.0 when thinking is enabled
   (Anthropic requires it)
2. **Match `max_tokens` exactly** — different values change `budget_tokens`,
   and `max_tokens` must be strictly > `budget_tokens` for thinking to activate
3. **Match `reasoning.effort`** — obviously, but worth stating explicitly

In our benchmark, the self-report now uses the same `temperature` and
`max_tokens` as the conversation turns.  An explicit `cache_control` breakpoint
is also placed on the last conversation message (before the questionnaire) so
the lookback finds the prior turn's cache entry.

---

## Automatic Caching Breakpoint Placement (Secondary Issue)

Even with matching parameters, automatic caching (`cache_control` at the
request body level) places its breakpoint on the **last cacheable block** of
each request.  In self-report requests, the last block is the questionnaire
(new content each time), so the automatic breakpoint never lands where the
prior conversation turn wrote its entry.

Fix: place an **explicit** `cache_control` breakpoint on the last conversation
message (the one right before the questionnaire).  This creates a lookback
starting point at the same position where the prior turn's automatic
breakpoint wrote.

See also: [Anthropic Prompt Caching docs — Explicit cache breakpoints](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

---

## Quick Reference: What Invalidates Anthropic Cache

From the official docs, plus our empirical findings:

| Change                                     | Tools Cache | System Cache | Messages Cache |
|--------------------------------------------|:-----------:|:------------:|:--------------:|
| Tool definitions                           | ✘          | ✘           | ✘             |
| Web search / citations toggle              | ✓          | ✘           | ✘             |
| Speed setting (`speed: "fast"` toggle)     | ✓          | ✘           | ✘             |
| Tool choice                                | ✓          | ✓           | ✘             |
| Images                                     | ✓          | ✓           | ✘             |
| **Thinking parameters (enable/disable/budget)** | **✓** | **✓**       | **✘**         |
| *`temperature` (if it toggles thinking)*   | *✓*        | *✓*         | *✘*           |
| *`max_tokens` (if it changes budget)*      | *✓*        | *✓*         | *✘*           |

*Italicized rows are our empirical findings — not documented by Anthropic but
confirmed through controlled experiments.*

---

## Key Anthropic Thinking Constraints

| Constraint | Details | Source |
|-----------|---------|--------|
| Temperature must be 1.0 | "Thinking isn't compatible with `temperature` or `top_k` modifications" | [Anthropic Extended Thinking docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) |
| `max_tokens > budget_tokens` | "max_tokens must be strictly higher than the reasoning budget" | [OpenRouter Reasoning Tokens docs](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) |
| Budget formula | `budget_tokens = max(min(max_tokens × effort_ratio, 128000), 1024)` | OpenRouter Reasoning Tokens docs |
| Effort ratios | xhigh=0.95, high=0.80, medium=0.50, **low=0.20**, minimal=0.10 | OpenRouter Reasoning Tokens docs |
| Min budget | 1024 tokens (floor) | Anthropic Extended Thinking docs |
| Silent disable | OpenRouter silently disables thinking instead of erroring when constraints aren't met | Empirical observation |
| Cache invalidation | Thinking on→off or off→on invalidates message-level cache | Anthropic Extended Thinking docs |
| Cache TTL | 5 minutes (default), 1 hour (at 2× cost) | Anthropic Prompt Caching docs |
| Min cacheable tokens | 4096 for Haiku 4.5, 1024 for Sonnet 4.5 | Anthropic Prompt Caching docs |

---

## Safe `max_tokens` Values for Reasoning

Given the formula `budget = max(min(max_tokens × ratio, 128000), 1024)` and the
constraint `max_tokens > budget`, here are minimum safe `max_tokens` values:

| Effort  | Ratio | Min `max_tokens` for thinking to activate |
|---------|-------|------------------------------------------|
| minimal | 0.10  | 1025 (budget=1024, need >1024)            |
| low     | 0.20  | 1025 (budget=1024, need >1024)            |
| medium  | 0.50  | 2049 (budget=1024, need >1024)            |
| high    | 0.80  | 5121 (budget=1024@low max, scales up)     |
| xhigh   | 0.95  | ≥1025 (budget=1024@low max)              |

**At `effort: "low"` with `max_tokens=1024`**: budget computes to
`max(204, 1024) = 1024`, so `max_tokens == budget` and thinking is disabled.
Use `max_tokens ≥ 1025` (or better, 2048+) to ensure thinking activates.

---

## Lookback Window Constraints

Automatic caching uses a **20-block lookback** window from the breakpoint
position.  For our benchmark:

- 36 conversation turns = ~74 messages + system = ~75 blocks
- After turn 36, the automatic breakpoint is at block 75
- The prior turn's entry is at block 73 (2 blocks back) — well within the 20-block window
- Self-report adds 2 more messages → block 77, lookback finds 75 → OK

If conversations grow beyond ~20 exchanges between checkpoints, the lookback
might miss older entries.  In that case, add a second explicit breakpoint
closer to the target position.

---

## References

1. [Anthropic — Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
2. [Anthropic — Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
3. [OpenRouter — Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
4. [AWS Bedrock — Extended Thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html)
5. [cline/cline#2712 — Temperature=1 requirement](https://github.com/cline/cline/issues/2712)
