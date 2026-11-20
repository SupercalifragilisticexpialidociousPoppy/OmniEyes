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


class EarlyLayerGate:
    def __init__(self, model, k=3, device="cuda"):
        self.visual = model.visual if hasattr(model, "visual") else model.model.visual
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
    def resume_from_k(self, early_feats, grid_thw):
        """Run blocks K..N + merger on pre-computed block-K features.
        
        Args:
            early_feats: tensor from early_features(), e.g. on CPU float32
            grid_thw: tensor [[t, h, w]]
        
        Returns:
            (final_embeddings, seconds)
        """
        import inspect
        import torch.nn.functional as F

        hidden = early_feats.to(self.device, self.dtype)
        grid_thw = grid_thw.to(self.device)

        # Prepare kwargs matching block signature
        block_kwargs = {}
        sig = inspect.signature(self.blocks[0].forward)

        rotary_pos_emb = None
        if hasattr(self.visual, "rot_pos_emb") and callable(self.visual.rot_pos_emb):
            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)

        cu_seqlens = None
        if "cu_seqlens" in sig.parameters:
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
            block_kwargs["cu_seqlens"] = cu_seqlens

        if "rotary_pos_emb" in sig.parameters and rotary_pos_emb is not None:
            block_kwargs["rotary_pos_emb"] = rotary_pos_emb

        if "max_seqlen" in sig.parameters and cu_seqlens is not None:
            if hasattr(self.visual, "compute_attn_mask_seqlen"):
                max_seqlen, seqlens = self.visual.compute_attn_mask_seqlen(cu_seqlens)
            else:
                max_seqlen = int(cu_seqlens.diff().max().item())
                seqlens = cu_seqlens.diff().tolist()
            block_kwargs["max_seqlen"] = max_seqlen
            if "seqlens" in sig.parameters:
                block_kwargs["seqlens"] = seqlens

        self._sync(); t0 = time.perf_counter()

        # Run remaining blocks (K through N-1)
        for blk in self.blocks[self.k:]:
            out = blk(hidden, **block_kwargs)
            hidden = out[0] if isinstance(out, tuple) else out

        # Run merger if present
        merger = getattr(self.visual, "merger", None)
        if merger is not None:
            final = merger(hidden)
        else:
            final = hidden

        self._sync(); dt = time.perf_counter() - t0
        return final.float().cpu(), dt


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
