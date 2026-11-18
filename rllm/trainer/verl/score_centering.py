"""Score centering for the opt-in verl/FSDP path.

The recorded head keeps its original sampler mass. Outside that head we model
q as proportional to the detached trainer tail. IS weights use the current
trainer / sampler ratio and are detached for BOTH the sampled and center terms.
See https://arxiv.org/abs/2609.20807 and verl's score_centering.py for the
head/tail derivation. This module does not implement PPO clipping.
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

SC_FIELDS = ("rollout_topk_ids", "rollout_topk_log_probs", "rollout_support_complete")


def candidate_log_probs(logits: torch.Tensor, ids: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Full-vocabulary normalized gathers; recompute fp32 chunks in backward.

    Checkpointing retains the original logits, not a full fp32 softmax. The
    workspace bound is chunk_size * vocabulary_size, plus the original logits
    and their gradient. This intentionally uses out-of-place operations.
    """
    if chunk_size < 1 or logits.ndim != 2 or ids.ndim != 2 or logits.shape[0] != ids.shape[0]:
        raise ValueError("Invalid logits/ids shape or chunk_size")

    def gather(chunk, indices):
        work = chunk.float()
        return work.gather(-1, indices.long()) - torch.logsumexp(work, dim=-1, keepdim=True)

    pieces = []
    for start in range(0, len(logits), chunk_size):
        chunk, indices = logits[start : start + chunk_size], ids[start : start + chunk_size]
        if torch.is_grad_enabled() and chunk.requires_grad:
            pieces.append(checkpoint(gather, chunk, indices, use_reentrant=False))
        else:
            pieces.append(gather(chunk, indices))
    return torch.cat(pieces) if pieces else logits.new_empty(ids.shape, dtype=torch.float32)


def centered_token_loss(sample_logp, sample_logq, head_logp, head_logq, complete, advantages, tis_cap=None):
    """Return per-token SC PG loss and head masses, without aggregation.

    Complete heads are evaluated on the verified sampler support. Incomplete
    heads use the trainer-tail approximation; head mass is never renormalized.
    Zero-probability candidate slots are padding and contribute no p or q mass.
    """
    if tis_cap is not None and (not math.isfinite(tis_cap) or tis_cap <= 0):
        raise ValueError("Score centering TIS cap must be finite and positive")
    if not torch.isfinite(sample_logq).all():
        raise ValueError("Sampled tokens require finite actual sampler log probabilities")
    head_logq = head_logq.detach().float()
    valid = torch.isfinite(head_logq)
    head_logp = head_logp.float()
    safe_logp = head_logp.masked_fill(~valid, 0)
    p = head_logp.exp().masked_fill(~valid, 0)
    q = head_logq.exp()
    p_mass, q_mass = p.sum(-1), q.sum(-1)
    if (q_mass > 1.0001).any() or (q_mass <= 0).any():
        raise ValueError("Invalid sampler head probability mass")
    if (complete & ((q_mass - 1).abs() > 1e-4)).any():
        raise ValueError("Verified complete sampler heads must have unit mass")

    def weight(log_ratio):
        if tis_cap is None:
            return torch.ones_like(log_ratio)
        return log_ratio.clamp(max=math.log(tis_cap)).exp()

    # Support-normalized branch. Dummy incomplete rows remain numerically valid.
    support_log_mass = head_logp.masked_fill(~valid, -torch.inf).logsumexp(-1)
    restricted_logp = safe_logp - support_log_mass.unsqueeze(-1)
    with torch.no_grad():
        weighted_q = q * weight(restricted_logp.detach() - head_logq).masked_fill(~valid, 0)
        support_residual = weighted_q - weighted_q.sum(-1, keepdim=True) * restricted_logp.exp().masked_fill(~valid, 0)
    support_center = (support_residual * restricted_logp).sum(-1)

    # Subtract alpha * E_p[score] = 0 to express the modeled tail using head entries only.
    with torch.no_grad():
        rho = (1 - q_mass).clamp_min(1e-6) / (1 - p_mass.detach()).clamp_min(1e-6)
        alpha = rho * weight(-rho.log())
        head_weight = weight(head_logp.detach() - head_logq).masked_fill(~valid, 0)
        tail_residual = q * head_weight - alpha.unsqueeze(-1) * p.detach()
    tail_center = (tail_residual * safe_logp).sum(-1)
    center = torch.where(complete, support_center, tail_center)
    scored_sample = sample_logp.float() - torch.where(complete, support_log_mass, 0)
    with torch.no_grad():
        sample_weight = weight(scored_sample.detach() - sample_logq.detach())
    loss = -advantages.detach() * (sample_weight * scored_sample - center)
    return loss, q_mass.detach(), p_mass.detach()


def validate_head(ids, log_probs, complete, response_length):
    """Validate persisted sampler heads before padding or cropping."""
    if log_probs is None or complete is None or len(ids) != response_length or len(log_probs) != response_length or len(complete) != response_length:
        raise ValueError("Score centering candidate fields must match response length")
    k = len(ids[0]) if ids else 0
    if response_length and not k:
        raise ValueError("Empty score centering candidate head")
    for token_ids, lps, is_complete in zip(ids, log_probs, complete, strict=True):
        if len(token_ids) != k or len(lps) != k or len(set(token_ids)) != k or any(i < 0 for i in token_ids):
            raise ValueError("Score centering heads require distinct nonnegative IDs and a fixed width")
        if any(math.isnan(lp) or lp > 0 for lp in lps):
            raise ValueError("Invalid sampler candidate log probabilities")
        mass = sum(math.exp(lp) for lp in lps)
        if mass <= 0 or mass > 1.0001 or (is_complete and abs(mass - 1) > 1e-4):
            raise ValueError("Invalid sampler head mass or completeness claim")
    return k


def batch_rollout_heads(heads, responses, max_length):
    if len(heads) != len(responses) or any(head is None for head in heads):
        raise ValueError("Missing score centering candidate distribution in batch")
    widths = [validate_head(*head, len(response)) for head, response in zip(heads, responses, strict=True)]
    k = max(widths, default=0)
    if not k:
        raise ValueError("Score centering batch has no response candidates")
    if any(width not in (0, k) for width in widths):
        raise ValueError("Mixed score centering candidate widths")
    ids = torch.arange(k, dtype=torch.int64).expand(len(heads), max_length, k).clone()
    lps = torch.full(ids.shape, -math.log(k), dtype=torch.float32)
    complete = torch.zeros((len(heads), max_length), dtype=torch.bool)
    for row, (head, response) in enumerate(zip(heads, responses, strict=True)):
        n = min(len(response), max_length)
        if n:
            ids[row, :n] = torch.tensor(head[0][:n], dtype=torch.int64)
            lps[row, :n] = torch.tensor(head[1][:n], dtype=torch.float32)
            complete[row, :n] = torch.tensor(head[2][:n], dtype=torch.bool)
    return dict(zip(SC_FIELDS, (ids, lps, complete), strict=True))


def heads_to_no_padding(data):
    """Move response heads to the rows that *predict* response tokens.

    Invoke before verl removes padding. Candidate tensors use full-sequence
    offsets and survive microbatch slicing/DP dispatch as nested tensors.
    """
    if not any(key in data for key in SC_FIELDS):
        return
    if not all(key in data for key in SC_FIELDS):
        raise ValueError("Incomplete score centering batch fields")
    mask, prompt_width = data["attention_mask"], data["prompts"].shape[1]
    k = data[SC_FIELDS[0]].shape[-1]
    rows = {key: [] for key in SC_FIELDS}
    for row in range(mask.shape[0]):
        p = int(mask[row, :prompt_width].sum())
        r = int(mask[row, prompt_width:].sum())
        if p < 1:
            raise ValueError("Score centering requires a nonempty prompt")
        ids = torch.arange(k, device=mask.device).expand(p + r, k).clone()
        lps = torch.full((p + r, k), -math.log(k), device=mask.device)
        complete = torch.zeros(p + r, dtype=torch.bool, device=mask.device)
        for key, value in zip(SC_FIELDS, (ids, lps, complete), strict=True):
            value[p - 1 : p - 1 + r] = data[key][row, :r]
            rows[key].append(value)
    for key in SC_FIELDS:
        data[key] = torch.nested.as_nested_tensor(rows[key], layout=torch.jagged)
