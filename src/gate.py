"""OmniEyes: early-layer variance gating -- core gate.

Runs only the first K blocks of the VLM's vision encoder on a frame (early exit
via a forward hook), returns block-K features, and selects frames whose
block-K features differ from the last *kept* frame (cosine sim < tau).
"""
import time
import numpy as np
import torch


class _EarlyExit(Exception):
    def __init__(self, out):
        self.out = out


def _vision_blocks(visual):
    # Qwen2-VL / Qwen2.5-VL: visual.blocks ; fall back to common names
    for name in ("blocks", "layers", "encoder.layers", "encoder.layer"):
        mod = visual
        try:
            for p in name.split("."):
                mod = getattr(mod, p)
            return mod
        except AttributeError:
            continue
    raise AttributeError("Cannot locate vision transformer blocks")


def _vision_encoder(model):
    # Qwen2-VL: model.visual or model.model.visual; Qwen2.5-Omni Thinker: model.visual;
    # full Qwen2.5-Omni: model.thinker.visual
    for path in ("visual", "model.visual", "thinker.visual", "thinker.model.visual"):
        mod = model
        try:
            for p in path.split("."):
                mod = getattr(mod, p)
            return mod
        except AttributeError:
            continue
    raise AttributeError("Cannot locate the vision encoder; print(model) to find it")


class EarlyLayerGate:
    def __init__(self, model, k=3, device="cuda"):
        self.visual = _vision_encoder(model)
        self.blocks = _vision_blocks(self.visual)
        self.n_blocks = len(self.blocks)
        assert 1 <= k <= self.n_blocks
        self.k = k
        self.device = device
        self.dtype = next(self.visual.parameters()).dtype

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def early_features(self, pixel_values, grid_thw):
        """Run blocks 1..K only. Returns (features [tokens, dim] float32 CPU, seconds)."""
        def hook(_m, _i, out):
            raise _EarlyExit(out[0] if isinstance(out, tuple) else out)
        h = self.blocks[self.k - 1].register_forward_hook(hook)
        self._sync(); t0 = time.perf_counter()
        try:
            self.visual(pixel_values.to(self.device, self.dtype), grid_thw=grid_thw.to(self.device))
            raise RuntimeError("early exit hook did not fire")
        except _EarlyExit as e:
            feats = e.out
        finally:
            h.remove()
        self._sync(); dt = time.perf_counter() - t0
        return feats.float().cpu(), dt

    @torch.no_grad()
    def full_time(self, pixel_values, grid_thw):
        """Time of the full (all-blocks + merger) vision forward, for cost accounting."""
        self._sync(); t0 = time.perf_counter()
        self.visual(pixel_values.to(self.device, self.dtype), grid_thw=grid_thw.to(self.device))
        self._sync()
        return time.perf_counter() - t0

    @torch.no_grad()
    def split_time(self, pixel_values, grid_thw):
        """Time one full vision pass, split at block K.

        Returns (full_seconds, resume_seconds), where resume_seconds is the time from the
        end of block K to the end of the pass (blocks K+1..N plus the merger): the "ideal"
        cost of a kept frame that could continue from its block-K features.
        A forward hook marks the clock when block K finishes, so the model runs its own
        code path (position embeddings, window attention) unchanged in any transformers version.
        """
        marks = {}

        def hook(_m, _i, _out):
            self._sync()
            marks["k"] = time.perf_counter()

        h = self.blocks[self.k - 1].register_forward_hook(hook)
        try:
            self._sync(); t0 = time.perf_counter()
            self.visual(pixel_values.to(self.device, self.dtype), grid_thw=grid_thw.to(self.device))
            self._sync(); t1 = time.perf_counter()
        finally:
            h.remove()
        if "k" not in marks:
            raise RuntimeError("block-K timing hook did not fire")
        return t1 - t0, t1 - marks["k"]


def token_cosine(a, b):
    """Mean per-token cosine similarity between two [T, D] feature maps (same T)."""
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return float((a * b).sum(-1).mean())


def select_by_gate(feats, tau):
    """Greedy streaming selection. feats: list of [T,D] tensors in time order.
    Frame 0 always kept. Frame i kept iff cos(feat_i, feat_last_kept) < tau."""
    kept, sims = [0], [1.0]
    ref = feats[0]
    for i in range(1, len(feats)):
        s = token_cosine(feats[i], ref)
        sims.append(s)
        if s < tau:
            kept.append(i)
            ref = feats[i]
    return kept, sims


def select_by_pixel_diff(frames_np, n_keep):
    """Cheap baseline gate matched to n_keep frames: frame 0 + the (n_keep-1) frames
    with the largest mean abs pixel change vs the previous frame."""
    if n_keep <= 1 or len(frames_np) <= 1:
        return [0]
    f = np.stack([x.astype(np.float32) / 255.0 for x in frames_np])
    diff = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    top = (np.argsort(-diff)[: n_keep - 1] + 1).tolist()
    return sorted(set([0] + top))


def select_uniform(n_total, n_keep):
    """Uniform subsampling baseline matched to n_keep frames (always includes last frame)."""
    n_keep = max(1, min(n_keep, n_total))
    idx = np.linspace(0, n_total - 1, n_keep).round().astype(int)
    return sorted(set(idx.tolist()))
