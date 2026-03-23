"""Quick analysis of step 9000 eval data — run after eval completes."""
import json, sys

with open("results/v060b_metrics.json") as f:
    data = json.load(f)

latest = data[-1]
step = latest["step"]
bpt = latest["test_bpt"]
d = latest.get("per_depth_bpt", {})
e = latest.get("per_pass_entropy", {})

print(f"\n{'='*60}")
print(f"STEP {step} EVAL RESULTS")
print(f"{'='*60}")
print(f"\nHeadline BPT (D=12): {bpt:.4f}")
print(f"Best BPT ever: {latest['best_bpt']:.4f}")
print(f"Is best: {latest.get('is_best', False)}")

if d:
    d8 = d.get("8", 99)
    d10 = d.get("10", 99)
    d12 = d.get("12", 99)
    print(f"\nPer-depth: D=1:{d.get('1',99):.4f}  D=4:{d.get('4',99):.4f}  "
          f"D=8:{d8:.4f}  D=10:{d10:.4f}  D=12:{d12:.4f}")
    print(f"D10-D12 gap: {d10-d12:+.4f} ({'D10 wins' if d10 < d12 else 'D12 wins'})")
    print(f"D8-D12 gap:  {d8-d12:+.4f} ({'D8 wins' if d8 < d12 else 'D12 wins'})")

    best_d = min(d.items(), key=lambda x: x[1])
    print(f"Optimal depth: D={best_d[0]} (BPT={best_d[1]:.4f})")

if e:
    ent_vals = [(int(k), v) for k, v in e.items()]
    ent_vals.sort()
    spread = max(v for _,v in ent_vals) / min(v for _,v in ent_vals)
    min_ent = min(ent_vals, key=lambda x: x[1])
    print(f"\nEntropy spread: {spread:.3f}x")
    print(f"Min entropy at pass {min_ent[0]}: {min_ent[1]:.4f}")

# Trough D8 analysis
print(f"\n--- Trough D8 Analysis ---")
post_restart = [(e_item["step"], e_item["per_depth_bpt"].get("8", 99))
                for e_item in data if e_item["step"] >= 4000 and "per_depth_bpt" in e_item]
for i, (s, d8_val) in enumerate(post_restart):
    prev = post_restart[i-1][1] if i > 0 else float('inf')
    nxt = post_restart[i+1][1] if i < len(post_restart)-1 else float('inf')
    if d8_val < prev and d8_val < nxt:
        print(f"Step {s}: Local trough D8 = {d8_val:.4f}")

# D10 vs D12 streak
d10_wins = sum(1 for e_item in data
               if "per_depth_bpt" in e_item
               and e_item["per_depth_bpt"].get("10", 99) < e_item["per_depth_bpt"].get("12", 99))
print(f"\nD10 beats D12: {d10_wins}/{len(data)} checkpoints ({d10_wins/len(data)*100:.0f}%)")

print(f"\n{'='*60}")
