# Dynamic Sharpening

**Per-token adaptive temperature for Transformer attention.**

## Key Result

| Context | Optimal Sharpness | PPL vs Baseline |
|---------|:---:|:---:|
| Simple text ("The cat sat on the mat") | **5×** | **+65%** |
| Complex text (technical/quantum) | **1×** | 0% |

**Different contexts need different attention sharpness.** Per-token adaptive temperature is justified.

## How It Works

Standard Transformer attention uses fixed scaling `1/√d`. Dynamic sharpening replaces it with `sharpness(entropy(context)) / √d` — lower entropy (uncertain context) → higher sharpness → more focused attention.

Synthetic test (test_synthetic.py):
- uncertain contexts → sharpness > 2.0 → +focus
- clear contexts → sharpness ≈ 1.0 → no change

Qwen test (test_qwen.py):
- simple texts optimal at 5× sharpness
- complex texts optimal at 1× — different contexts confirmed

Per-token test (test_per_token.py, GPT-2):
- simple text entropy: 1.01
- technical text entropy: 1.50
- ending text entropy: 1.80
→ per-token variation proven

## Quick Start

```bash
python test_synthetic.py    # synthetic entropy test
python test_qwen.py         # Qwen-0.5B sharpness grid search
python test_per_token.py    # GPT-2 per-token entropy measurement
```

## License

MIT
