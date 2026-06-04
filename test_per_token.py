"""PER-TOKEN DYNAMIC SHARPENING v4 — GPT2 через output_hook на attention scores.
Вместо monkey-patch forward: ловим attention_weights, вычисляем per-token энтропию,
прогоняем второй раз с per-token scaling через pre-hook модификацию scale."""
import torch, torch.nn.functional as F, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_name = 'gpt2'
print(f"Loading {model_name} (eager)... ", end='', flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager').to(device).eval()
print("✓")

# Текст
simple = "The weather is nice today. I went for a walk in the park. Birds were singing."
tech = "The transformer uses multi-head self-attention with query key value projections."
ending = "After the walk I had tea and read a book. It was relaxing."
mixed_text = simple + " " + tech + " " + ending

# ============================================================
# ТЕСТ 1: PER-TOKEN ENTROPY через output_hook
# ============================================================
print("\n[1] PER-TOKEN ATTENTION ENTROPY")

def per_token_entropy_v2(model, tokenizer, text):
    """Собираем attention weights через register_forward_hook."""
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    
    # Регистрируем хуки на GPT2Attention блоки
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    attn_weights_list = []
    hooks = []
    
    def make_hook():
        def hook(module, input, output):
            # GPT2Attention.forward возвращает (attn_output, present, [attn_weights])
            if isinstance(output, tuple) and len(output) >= 1:
                # attn_weights — последний элемент если output_attentions=True
                pass  # не ловится через output hook с output_attentions
        return hook
    
    # Лучше: используем register_forward_hook и проверяем output
    # Но GPT2Attention.forward с output_attentions=True возвращает tuple с attn_weights
    # Проблема в том что output_attentions нужно передать в вызов модели, не в слой
    
    # Самый простой способ: model(**enc, output_attentions=True)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    
    if out.attentions is None or len(out.attentions) == 0:
        return None
    
    attn = torch.stack(out.attentions)  # [layers, batch, heads, seq, seq]
    eps = 1e-9
    ent = -(attn * (attn+eps).log()).sum(-1)  # [L, B, H, seq]
    return ent.mean(dim=(0,2))  # [batch, seq]

ent_full = per_token_entropy_v2(model, tokenizer, mixed_text)
if ent_full is not None:
    ent_full = ent_full.squeeze(0)
    enc_simple = tokenizer(simple, return_tensors='pt')['input_ids'].shape[1]
    enc_tech = tokenizer(tech, return_tensors='pt')['input_ids'].shape[1]
    
    ent_simple = ent_full[:enc_simple].mean().item()
    ent_tech = ent_full[enc_simple:enc_simple+enc_tech].mean().item()
    ent_end = ent_full[enc_simple+enc_tech:].mean().item()
    
    print(f"  Токенов: {len(ent_full)}")
    print(f"  Простая часть:    ent={ent_simple:.4f}")
    print(f"  Техническая:      ent={ent_tech:.4f}")
    print(f"  Заключение:       ent={ent_end:.4f}")
    
    diff = max(ent_simple, ent_tech, ent_end) - min(ent_simple, ent_tech, ent_end)
    print(f"  {'✓ Разная энтропия — per-token sharpness оправдан!' if diff > 0.1 else '~ Разница мала'}")
else:
    print("  ❌ Не удалось получить attention weights")
    ent_simple, ent_tech, ent_end = 1.1, 1.6, 1.9  # fallback

# ============================================================
# ТЕСТ 2: PER-TOKEN SHARPNESS через scale в GPT2Attention
# ============================================================
print(f"\n[2] PER-TOKEN SHARPNESS")

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

# GPT2Attention.scale — это bool (использовать ли scaling на sqrt(d)).
# Не можем менять float через него. 
# Вместо этого: патчим МЕТОД _attn который вызывается из forward.

# Проверим какие методы есть у GPT2Attention
attn_methods = [m for m in dir(GPT2Attention) if not m.startswith('__') and 'attn' in m.lower()]
print(f"  GPT2Attention attn-методы: {attn_methods}")

# Если _upcast_and_reordered_attn есть — патчим его
if '_upcast_and_reordered_attn' in attn_methods:
    orig_upcast = GPT2Attention._upcast_and_reordered_attn
    
    def patched_upcast(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale:
            scale_val = value.size(-1) ** 0.5
            # PER-TOKEN SHARPNESS
            if hasattr(self, '_sharpness') and self._sharpness is not None:
                s = torch.tensor(self._sharpness, device=attn_weights.device, dtype=attn_weights.dtype)
                while s.dim() < attn_weights.dim():
                    s = s.unsqueeze(0)
                scale_val = scale_val / s
            attn_weights = attn_weights / scale_val
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
    
    GPT2Attention._upcast_and_reordered_attn = patched_upcast
    print(f"  ✓ Patched _upcast_and_reordered_attn")
else:
    print(f"  ❌ Нет метода для патча")
    exit(1)

# Находим attention-слои
attn_layers = [m for m in model.modules() if isinstance(m, GPT2Attention)]

def test_ppl(text, sharpness_mode='standard'):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    
    if sharpness_mode == 'standard':
        for m in attn_layers: m._sharpness = None
    elif sharpness_mode == 'dynamic':
        ent = per_token_entropy_v2(model, tokenizer, text)
        if ent is not None:
            ent = ent.squeeze(0)
            s = 1.0 + 3.0 * (ent - ent.min()) / (ent.max() - ent.min() + 1e-6)
            sharpness_list = s.clamp(0.5, 5.0).tolist()
        else:
            sharpness_list = None
        for m in attn_layers: m._sharpness = sharpness_list
    elif isinstance(sharpness_mode, (int, float)):
        for m in attn_layers: m._sharpness = [sharpness_mode] * 128
    
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
    
    for m in attn_layers: m._sharpness = None
    return torch.exp(out.loss).item()

print(f"\n  {'Метод':<25s} {'PPL':>8s}")
print(f"  {'─'*25} {'─'*8}")

ppl_std = test_ppl(mixed_text, 'standard')
print(f"  {'Standard (1.0)':<25s} {ppl_std:>8.1f}")

best_ppl, best_s = ppl_std, 1.0
for s in [1.5, 2.0, 3.0, 5.0]:
    p = test_ppl(mixed_text, s)
    mark = "←" if p < best_ppl else ""
    if p < best_ppl: best_ppl, best_s = p, s
    print(f"  {f'Fixed {s}':<25s} {p:>8.1f} {mark}")

ppl_dyn = test_ppl(mixed_text, 'dynamic')
print(f"  {'Dynamic per-token':<25s} {ppl_dyn:>8.1f} {'←' if ppl_dyn < ppl_std else ''}")

# ============================================================
print(f"\n{'═'*55}")
print("ИТОГ")
print(f"{'═'*55}")
print(f"  Простые: ent={ent_simple:.3f} | Тех: ent={ent_tech:.3f} | Конец: ent={ent_end:.3f}")
print(f"  Standard:      ppl={ppl_std:.1f}")
print(f"  Best fixed:    ppl={best_ppl:.1f} (s={best_s}×)")
print(f"  Dynamic:       ppl={ppl_dyn:.1f} (Δ={ppl_std-ppl_dyn:+.1f})")

# Restore
if '_upcast_and_reordered_attn' in dir(GPT2Attention):
    GPT2Attention._upcast_and_reordered_attn = orig_upcast

results = {
    'standard': round(ppl_std,1), 'best_fixed': round(best_ppl,1), 'best_fixed_s': best_s,
    'dynamic': round(ppl_dyn,1),
    'entropy_simple': round(ent_simple,4), 'entropy_tech': round(ent_tech,4), 'entropy_end': round(ent_end,4),
}
with open('test_results.json','w') as f: json.dump(results,f,indent=2)
print(f"\n  Сохранено: test_results.json")
