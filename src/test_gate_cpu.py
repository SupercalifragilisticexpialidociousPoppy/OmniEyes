"""CPU smoke test: tiny random Qwen2-VL vision tower, checks early exit, split timing, and selection logic."""
import numpy as np, torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from gate import EarlyLayerGate, select_by_gate, select_uniform, select_by_pixel_diff

cfg = Qwen2VLConfig(vision_config=dict(depth=6, embed_dim=64, hidden_size=64, num_heads=4, mlp_ratio=2),
                    hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=1000, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]})
m = Qwen2VLForConditionalGeneration(cfg).eval()
g = EarlyLayerGate(m, k=2, device="cpu")
print("blocks", g.n_blocks)
grid = torch.tensor([[1, 8, 8]])
pv = torch.randn(64, 3 * 2 * 14 * 14)
f1, t1 = g.early_features(pv, grid)
f2, _ = g.early_features(pv + 0.01 * torch.randn_like(pv), grid)
f3, _ = g.early_features(torch.randn_like(pv), grid)
print("feat", f1.shape, "gate s", round(t1, 4), "full s", round(g.full_time(pv, grid), 4))

# Split timing: one real full pass, clock marked when block K finishes
t_full, t_resume = g.split_time(pv, grid)
print("split: full s", round(t_full, 4), "resume s", round(t_resume, 4))
assert 0 < t_resume < t_full, (t_full, t_resume)
print("split_time assertion OK")

kept, sims = select_by_gate([f1, f2, f3], 0.95)
print("sims", [round(s, 3) for s in sims], "kept", kept)
assert kept == [0, 2], kept
print("uniform", select_uniform(33, 5))
fr = [np.zeros((4, 4, 3), np.uint8)] * 3 + [np.full((4, 4, 3), 255, np.uint8)]
print("pixel", select_by_pixel_diff(fr, 2))
print("ALL TESTS PASSED")
