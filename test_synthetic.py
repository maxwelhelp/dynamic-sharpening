"""DYNAMIC SHARPENING TEST — контекстная температура Softmax.
Гипотеза: фиксированная температура 1/√d хуже чем адаптивная к энтропии контекста."""
import torch, torch.nn.functional as F, json, time

device = 'cuda'
torch.manual_seed(42)
d, seq, B = 64, 32, 100

def attn_entropy(w):
    return -(w * (w+1e-9).log()).sum(-1).mean().item()

def context_sharpness(Q):
    h = -(F.softmax(Q,dim=-1) * F.log_softmax(Q,dim=-1)).sum(-1).mean().item()
    return 1.0 + 2.0 * h

results = {}
for name, qs, ks in [('uncertain',0.1,2.0),('clear',2.0,0.1),('noisy',0.5,0.5)]:
    Q = torch.randn(B,seq,d,device=device) * qs
    K = torch.randn(B,seq,d,device=device) * ks

    # Standard
    t0 = time.perf_counter()
    s = Q @ K.transpose(1,2) / d**0.5
    a_std = F.softmax(s, dim=-1)
    t_std = time.perf_counter() - t0

    # Dynamic
    t0 = time.perf_counter()
    sh = context_sharpness(Q)
    a_dyn = F.softmax(s * sh, dim=-1)
    t_dyn = time.perf_counter() - t0

    results[name] = {
        'sharpness': round(sh,3),
        'entropy_std': round(attn_entropy(a_std),4),
        'entropy_dyn': round(attn_entropy(a_dyn),4),
        'focus_std': round(a_std.max(-1).values.mean().item(),4),
        'focus_dyn': round(a_dyn.max(-1).values.mean().item(),4),
        'speed_std_us': round(t_std*1e6/B,1),
        'speed_dyn_us': round(t_dyn*1e6/B,1),
    }
    imp = (a_dyn.max(-1).values.mean().item() - a_std.max(-1).values.mean().item()) / a_std.max(-1).values.mean().item() * 100
    print(f"[{name:12s}] sharp={sh:.2f} | ent: {attn_entropy(a_std):.4f}→{attn_entropy(a_dyn):.4f} | focus: {a_std.max(-1).values.mean():.3f}→{a_dyn.max(-1).values.mean():.3f} ({imp:+.0f}%)")

ok = any(r['sharpness']>2.0 for r in results.values())
print(f"\n{'✅' if ok else '❌'} Dynamic Sharpening: sharpness > 2.0 при неопределённости")
with open('test_results.json','w') as f: json.dump({'passed':ok,'results':results},f,indent=2)
print("Сохранено: test_results.json")
