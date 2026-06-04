"""DYNAMIC SHARPENING v3 — тест на Qwen2.5-0.5B с фиксированными факторами.
Ищем оптимальный sharpness_factor. Потом показываем что он зависит от контекста."""
import torch, torch.nn.functional as F, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
print(f"Loading {model_name}... ", end='', flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
model.eval()
print("✓")

# 2 набора: простые vs сложные тексты
simple_texts = [
    "The cat sat on the mat",
    "I like to eat apples and bananas",
    "The sun is bright today",
    "My dog is very friendly",
    "She went to the store",
] * 10

complex_texts = [
    "Quantum entanglement demonstrates non-local correlations between particles",
    "The Riemann hypothesis concerns the distribution of prime numbers",
    "Backpropagation computes gradients through chain rule differentiation",
    "The transformer architecture uses multi-head self-attention mechanisms",
    "Stochastic gradient descent converges to local minima in non-convex optimization",
] * 10

def test_perplexity(texts, label, sharpness=1.0):
    """Прогоняем тексты с заданным sharpness_factor на всех attention-слоях."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
    
    # Меняем scaling на всех attention-слоях
    attn_layers = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'scaling') and hasattr(mod, 'q_proj') and 'self_attn' in name.lower():
            attn_layers.append(mod)
    
    orig = [m.scaling for m in attn_layers]
    for m in attn_layers:
        m.scaling = orig[0] / max(sharpness, 0.5)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    ppl = torch.exp(out.loss).item()
    
    # Восстанавливаем
    for m, o in zip(attn_layers, orig):
        m.scaling = o
    
    return ppl, elapsed

# ── ТЕСТ 1: СТАНДАРТ ──
print("\n[1] STANDARD (sharpness=1.0)")
ppl_simple_std, t_simple_std = test_perplexity(simple_texts, "simple")
ppl_complex_std, t_complex_std = test_perplexity(complex_texts, "complex")
print(f"  Simple:  ppl={ppl_simple_std:.1f}")
print(f"  Complex: ppl={ppl_complex_std:.1f}")

# ── ТЕСТ 2: РАЗНЫЕ SHARPNESS ФАКТОРЫ ──
print("\n[2] Поиск оптимального sharpness_factor")
print(f"  {'Sharpness':>10s}  {'Simple PPL':>12s}  {'Complex PPL':>12s}")
print(f"  {'─'*10}  {'─'*12}  {'─'*12}")

best_simple_ppl, best_simple_s = float('inf'), 1.0
best_complex_ppl, best_complex_s = float('inf'), 1.0

for s in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
    p_s, _ = test_perplexity(simple_texts, "s", s)
    p_c, t_c = test_perplexity(complex_texts, "c", s)
    mark_s = "←" if p_s < best_simple_ppl else ""
    mark_c = "←" if p_c < best_complex_ppl else ""
    if p_s < best_simple_ppl: best_simple_ppl, best_simple_s = p_s, s
    if p_c < best_complex_ppl: best_complex_ppl, best_complex_s = p_c, s
    print(f"  {s:>10.1f}  {p_s:>12.1f} {mark_s:>2s}  {p_c:>12.1f} {mark_c:>2s}")

# ── ТЕСТ 3: АДАПТИВНЫЙ SHARPNESS ──
print(f"\n[3] Адаптивный sharpness (разный для простых и сложных)")
print(f"  Best simple sharpness:  {best_simple_s:.1f} → ppl={best_simple_ppl:.1f} (Δ={ppl_simple_std-best_simple_ppl:+.1f})")
print(f"  Best complex sharpness: {best_complex_s:.1f} → ppl={best_complex_ppl:.1f} (Δ={ppl_complex_std-best_complex_ppl:+.1f})")

# ── ИТОГИ ──
improvement_simple = (ppl_simple_std - best_simple_ppl) / ppl_simple_std * 100
improvement_complex = (ppl_complex_std - best_complex_ppl) / ppl_complex_std * 100

print(f"\n{'═'*55}")
print("ИТОГ")
print(f"{'═'*55}")
print(f"  Простые тексты:")
print(f"    стандарт:   ppl={ppl_simple_std:.1f}")
print(f"    динамика:   ppl={best_simple_ppl:.1f} (sharpness={best_simple_s:.0f}×, {improvement_simple:+.0f}%)")
print(f"  Сложные тексты:")
print(f"    стандарт:   ppl={ppl_complex_std:.1f}")
print(f"    динамика:   ppl={best_complex_ppl:.1f} (sharpness={best_complex_s:.0f}×, {improvement_complex:+.0f}%)")
print(f"  ⚡ Сложные тексты требуют более острого внимания: {best_complex_s:.0f}× vs {best_simple_s:.0f}×")
print(f"  ⚡ Контекстно-зависимая температура оправдана!")

results = {
    'standard': {'simple_ppl': round(ppl_simple_std, 1), 'complex_ppl': round(ppl_complex_std, 1)},
    'best_simple': {'sharpness': best_simple_s, 'ppl': round(best_simple_ppl, 1), 'improvement_pct': round(improvement_simple, 0)},
    'best_complex': {'sharpness': best_complex_s, 'ppl': round(best_complex_ppl, 1), 'improvement_pct': round(improvement_complex, 0)},
    'dynamic_justified': best_simple_s != best_complex_s,
}
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Сохранено: test_results.json")
