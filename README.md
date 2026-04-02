# Persona Decay Bench

Benchmark for LLM persona decay: measures how well AI models sustain assigned persona expression over extended multi-turn conversations.

Replicates findings from the "Stable Personas" paper (arXiv:2601.22812v1), which discovered that while LLMs produce stable self-reports throughout extended conversations, **observer-rated persona expression declines** — especially for high-intensity ADHD conditions.

## What it measures

- **Initial Expression** — Can the model express a high-intensity ADHD persona convincingly?
- **Decay Resistance** — Does persona expression hold up over 24 conversation turns?
- **Self-Report Consistency** — Are the model's self-assessments stable across checkpoints?
- **Observer-Self Agreement** — Do self-reports match external observer ratings?
- **Extended Stability** — Does the model maintain persona beyond the paper's 18-turn window?

## Quick start

```bash
# Install
pip install -e ".[test]"

# Set your OpenRouter API key
cp .env.example .env
# Edit .env and add your key

# Run benchmark on specific models
persona-decay run --models "google/gemini-3-flash-preview"

# Evaluate observer ratings
persona-decay evaluate --models "google/gemini-3-flash-preview"

# Display leaderboard
persona-decay leaderboard

# Generate full report
persona-decay generate-report

# Estimate cost without running
persona-decay estimate-cost
```

## Methodology

Each model receives a high-intensity ADHD persona prompt and engages in a 24-turn conversation with a neutral partner. At 6-turn checkpoints (turns 6, 12, 18, 24), the model completes a 12-item ADHD self-assessment and is rated by an observer LLM.

The **Persona Stability Index** (0-100) is computed from weighted dimensions:
- Initial Expression: 20%
- Decay Resistance: 40%
- Self-Report Consistency: 15%
- Observer-Self Agreement: 10%
- Extended Stability: 15%

## License

MIT
