"""Component 1: expert capacity gating for speculative verification of an offloaded MoE.

`MoESpeculativeGater` decides, for ONE MoE layer of ONE verification block, which experts are allowed to run (and so which
experts may cross the PCIe bus).  It is the training-free selector of AcceptMoE (arXiv 2608.02989) with the reuse /
confidence ideas of S2-MoE (2608.15018) and the hard capacity of MoE-Spec (2602.16052), arranged from what was measured in
Shadow (docs/SPECULATION.md): substitution routing (each token still takes k experts, re-picked inside the set), the anchor
position routes naturally, positions are weighted by how likely they are to be committed, and the set is sized by the
effective rank of the demand under a HARD cap `k_max_experts`.

Notation: B sequences, T block positions (position 0 = the anchor, a token the target is certain to commit; positions
1..T-1 = drafted tokens), E experts, k the target's top-k, V the draft vocabulary.

Inputs
    draft_logits         [B, T, V]  the draft model's next-token logits at each row (row t proposes token t + 1)
    draft_routing_scores [B, T, E]  the draft-side route predictor's scores for this layer (logits; softmaxed here)
    commit_probs         [T] or [B, T]  offline MARGINAL commitment probabilities p_d = P(position d reaches the output);
                         p_0 is forced to 1.  `estimate_commit_probs` makes them from logged accepted lengths.
    target_router_prior  [B, T, E], [T, E] or [E], optional   probabilities from the target router (exact router
                         probabilities for the anchor, a running per-expert frequency for the rest)
    resident_mask        [E] or [B, E] bool, optional   True where the expert's weights are already in GPU memory

Output (`GateOutput`)
    expert_mask   [B, T, E] bool  the experts each position may be routed to.  Position 0 keeps its own natural top-k;
                  positions 1.. share the block's eligible set.
    transfer_mask [B, E] bool     expert_mask.any(dim=1): what may be fetched over PCIe for this layer.
    set_size      [B] long        |eligible set| (<= k_max_experts, >= k).

Selection, step by step (shape after each step)
    1. acceptance estimate per position: c_t = max softmax(draft_logits[:, t-1]) for t >= 1, c_0 = 1; prefix P_t =
       prod_{j<=t} c_j                                                        [B, T]
       alpha_t = p_t**beta * P_t**gamma  (offline commitment x draft confidence)  [B, T]
    2. routing demand: q = softmax(draft_routing_scores); with a target prior pi the two are fused as a geometric mixture
       r  ~  q**mix * pi**(1 - mix)  (renormalised); the anchor row uses the target prior alone   [B, T, E]
    3. natural top-k of r per token, weights renormalised over the k          idx [B, T, k], w [B, T, k]
       sparse [B, T, E] holds w at idx, 0 elsewhere; nat_mask [B, T, E] its support; votes [B, E] = #tokens naming e
    4. utility u_e = sum_t alpha_t * sparse[t, e]                               [B, E]
    5. anchor set A = nat_mask[:, 0] (never pruned); residual demand u_R = u outside A
    6. size n_er = ceil(size_scale * exp(H(u_R / sum u_R))) -- the effective rank (exponential of the entropy) -- clipped to
       the capacity k_max_experts - |A| and to the number of experts with demand; the n_er highest-utility residual
       experts (ties to the lower index) join A                                 S [B, E]
    7. residency-aware pruning (if resident_mask): the non-resident members of S \\ A are removed lowest utility first,
       while the natural assignments displaced by the removals stay within `rerouting_budget` x the assignments the
       set formation already displaced; anchors are never removed, so |S| >= k always.
    8. expert_mask = S broadcast over T, position 0 replaced by its natural top-k.
    With `shared_across_batch=True` the utilities are summed over the B sequences first so ONE set serves the whole batch
    (the hard cap then bounds the unique experts loaded per layer for the batch).  The sequences' anchors are exact and
    always run, so the set is the union of the anchors plus at most `k_max_experts - |union|` more: it is <= k_max_experts
    unless the anchors alone are more than that (at most B * k experts).

`route_with_mask` then does the substitution routing: TopK(z + mask) with the target's own softmax-over-the-k weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GateOutput:
    expert_mask: torch.Tensor    # [B, T, E] bool
    transfer_mask: torch.Tensor  # [B, E] bool
    set_size: torch.Tensor       # [B] long
    utility: torch.Tensor        # [B, E] float32 (commitment-weighted demand; broadcast over B when shared)
    anchor_mask: torch.Tensor    # [B, E] bool
    pruned_mask: torch.Tensor    # [B, E] bool, non-resident experts removed by residency-aware pruning


class MoESpeculativeGater(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        k_max_experts: int,
        beta: float = 1.0,
        gamma: float = 0.5,
        mix: float = 0.5,
        size_scale: float = 1.0,
        rerouting_budget: float = 1.0,
        shared_across_batch: bool = False,
        eps: float = 1e-9,
    ) -> None:
        """
        num_experts          E
        top_k                k, the target router's experts per token (2 for Mixtral)
        k_max_experts        hard cap on the eligible set per layer (per sequence, or per batch if shared); >= top_k
        beta                 sharpness of the offline commitment weights (the paper uses 0.5; Shadow measured 1.0 better)
        gamma                weight of the draft's own prefix confidence (0: offline commitment only)
        mix                  weight of the draft route predictor against the target prior in the demand (1: draft only)
        size_scale           multiplies the effective-rank set size (<1 tightens, >1 loosens)
        rerouting_budget     residency pruning may displace this multiple of the assignments set formation displaced
        shared_across_batch  one set for the whole batch instead of one per sequence
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k={top_k} must be in [1, {num_experts}]")
        if k_max_experts < top_k:
            raise ValueError(f"k_max_experts={k_max_experts} must be at least top_k={top_k} (the anchor needs its k experts)")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        self.E, self.k, self.k_max = num_experts, top_k, min(k_max_experts, num_experts)
        self.beta, self.gamma, self.mix = beta, gamma, mix
        self.size_scale, self.rerouting_budget = size_scale, rerouting_budget
        self.shared, self.eps = shared_across_batch, eps

    # ----------------------------------------------------------------------------------------------- offline statistics
    @staticmethod
    def estimate_commit_probs(accepted_lengths: torch.Tensor, block_len: int) -> torch.Tensor:
        """Marginal commitment probabilities from logged verification traces.

        accepted_lengths [N] long: for each logged round, how many drafted tokens were committed (0 .. block_len - 1).
        Returns [block_len] float: p_0 = 1 (the anchor), p_d = fraction of rounds with accepted_lengths >= d (d >= 1).
        This is the MARGINAL probability that a node at position d reaches the output, not the conditional acceptance rate."""
        d = torch.arange(block_len, device=accepted_lengths.device)
        p = (accepted_lengths.unsqueeze(1) >= d.unsqueeze(0)).float().mean(dim=0)  # [N, T] -> [T]
        p[0] = 1.0
        return p

    # ------------------------------------------------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(
        self,
        draft_logits: torch.Tensor,
        draft_routing_scores: torch.Tensor,
        commit_probs: torch.Tensor,
        target_router_prior: Optional[torch.Tensor] = None,
        resident_mask: Optional[torch.Tensor] = None,
    ) -> GateOutput:
        B, T, _ = draft_logits.shape
        E, k = self.E, self.k
        dev = draft_logits.device
        if draft_routing_scores.shape != (B, T, E):
            raise ValueError(f"draft_routing_scores must be [B={B}, T={T}, E={E}], got {tuple(draft_routing_scores.shape)}")

        # 1. per-position commitment weight alpha [B, T]
        conf = F.softmax(draft_logits.float(), dim=-1).amax(dim=-1)                     # [B, T]: confidence of row t's proposal
        step_conf = torch.cat([torch.ones(B, 1, device=dev), conf[:, : T - 1]], dim=1)  # [B, T]: c_0 = 1, c_t = conf[t - 1]
        prefix = torch.cumprod(step_conf, dim=1)                                        # [B, T]
        commit = commit_probs.float().to(dev)
        commit = commit.unsqueeze(0).expand(B, T) if commit.dim() == 1 else commit      # [B, T]
        commit = torch.cat([torch.ones(B, 1, device=dev), commit[:, 1:]], dim=1)         # the anchor is certain
        alpha = commit.clamp_min(self.eps).pow(self.beta) * prefix.clamp_min(self.eps).pow(self.gamma)  # [B, T]

        # 2. routing demand r [B, T, E]
        q = F.softmax(draft_routing_scores.float(), dim=-1)                              # [B, T, E]
        if target_router_prior is None:
            r = q
        else:
            pi = target_router_prior.float().to(dev)
            if pi.dim() == 1:
                pi = pi.view(1, 1, E).expand(B, T, E)
            elif pi.dim() == 2:
                pi = pi.unsqueeze(0).expand(B, T, E)
            pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(self.eps)                   # [B, T, E]
            log_r = self.mix * torch.log(q + self.eps) + (1.0 - self.mix) * torch.log(pi + self.eps)
            r = F.softmax(log_r, dim=-1)                                                 # geometric mixture, renormalised
            r = torch.cat([pi[:, :1], r[:, 1:]], dim=1)                                  # the anchor row: the target alone

        # 3. natural top-k per token: idx/w [B, T, k]; sparse and nat_mask [B, T, E]; votes [B, E]
        top_v, idx = r.topk(k, dim=-1)
        w = top_v / top_v.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        sparse = torch.zeros(B, T, E, device=dev).scatter(-1, idx, w)
        nat_mask = torch.zeros(B, T, E, dtype=torch.bool, device=dev).scatter(-1, idx, True)
        votes = nat_mask.sum(dim=1)                                                      # [B, E] long

        # 4. utility [B, E]
        utility = torch.einsum("bt,bte->be", alpha, sparse)

        # 5-7. select on [Bs, E] rows (Bs = B, or 1 when one set serves the batch)
        if self.shared:
            sel_u = utility.sum(dim=0, keepdim=True)
            sel_votes = votes.sum(dim=0, keepdim=True)
            anchor = nat_mask[:, 0].any(dim=0, keepdim=True)
        else:
            sel_u, sel_votes, anchor = utility, votes, nat_mask[:, 0]
        Bs = sel_u.shape[0]
        res = None
        if resident_mask is not None:
            res = resident_mask.to(dev).bool()
            res = res.view(1, E).expand(Bs, E) if res.dim() == 1 else (res.all(dim=0, keepdim=True) if self.shared else res)

        arange_e = torch.arange(E, device=dev).unsqueeze(0).expand(Bs, E)
        n_anchor = anchor.sum(dim=-1)                                                    # [Bs]
        u_r = sel_u.masked_fill(anchor, 0.0)                                             # residual demand
        z = u_r.sum(dim=-1, keepdim=True)
        p_r = u_r / z.clamp_min(self.eps)
        entropy = -(p_r * torch.log(p_r.clamp_min(self.eps))).sum(dim=-1)                # [Bs]
        n_er = torch.ceil(self.size_scale * torch.exp(entropy)).long()
        n_er = torch.where(z.squeeze(-1) > 0, n_er, torch.zeros_like(n_er))
        capacity = (self.k_max - n_anchor).clamp_min(0)
        n_take = torch.minimum(torch.minimum(n_er, capacity), (u_r > 0).sum(dim=-1))      # [Bs]
        order = torch.sort(u_r, dim=-1, descending=True, stable=True).indices            # [Bs, E] expert ids, best first
        rank = torch.empty_like(order).scatter(-1, order, arange_e)                      # rank[b, e] = position of e in order
        taken = rank < n_take.unsqueeze(-1)                                              # [Bs, E]
        eligible = anchor | taken

        pruned = torch.zeros_like(eligible)
        if res is not None:
            cand = taken & ~res                                                          # never the anchors
            displaced = (sel_votes * ((sel_votes > 0) & ~eligible)).sum(dim=-1)          # assignments formation displaced
            budget = torch.ceil(self.rerouting_budget * displaced.float()).long()        # [Bs]
            key = sel_u.masked_fill(~cand, float("inf"))
            order2 = torch.sort(key, dim=-1, descending=False, stable=True).indices      # candidates first, lowest utility first
            cand_sorted = cand.gather(-1, order2)
            votes_sorted = (sel_votes * cand).gather(-1, order2)
            cum = votes_sorted.cumsum(dim=-1)                                            # assignments displaced so far
            max_removals = (eligible.sum(dim=-1) - k).clamp_min(0)                       # keep |S| >= k
            rm_sorted = cand_sorted & (cum <= budget.unsqueeze(-1)) & (arange_e < max_removals.unsqueeze(-1))
            pruned = torch.zeros_like(eligible).scatter(-1, order2, rm_sorted)
            eligible = eligible & ~pruned

        # 8. masks per position / per sequence
        if self.shared:
            eligible = eligible.expand(B, E)
            pruned = pruned.expand(B, E)
            anchor_out = anchor.expand(B, E)
            sel_u = sel_u.expand(B, E)
        else:
            anchor_out = anchor
        expert_mask = eligible.unsqueeze(1).expand(B, T, E).clone()
        expert_mask[:, 0] = nat_mask[:, 0]                                               # the anchor routes naturally
        return GateOutput(
            expert_mask=expert_mask,
            transfer_mask=expert_mask.any(dim=1),
            set_size=eligible.sum(dim=-1),
            utility=sel_u,
            anchor_mask=anchor_out,
            pruned_mask=pruned,
        )

    # --------------------------------------------------------------------------------------------- substitution routing
    @staticmethod
    def route_with_mask(router_logits: torch.Tensor, expert_mask: torch.Tensor, top_k: int):
        """Substitution routing: TopK(z + M) where M = 0 for allowed experts and -inf otherwise.

        router_logits [B, T, E] (the target router's own logits), expert_mask [B, T, E] bool
        -> expert indices [B, T, k] long, mixing weights [B, T, k] float (softmax over the k chosen logits, which equals the
        target's softmax-then-renormalise-over-top-k).  Every token receives exactly k experts as long as the mask allows
        at least k, which the gater guarantees (|S| >= k and the anchor keeps its natural k)."""
        masked = router_logits.float().masked_fill(~expert_mask, float("-inf"))
        top_v, idx = masked.topk(top_k, dim=-1)
        return idx, F.softmax(top_v, dim=-1)


# ---------------------------------------------------------------------------------------------------------------- self-test
def _self_test() -> None:
    torch.manual_seed(0)
    B, T, V, E, k, kmax = 3, 5, 100, 32, 2, 7
    draft_logits = torch.randn(B, T, V) * 3
    draft_scores = torch.randn(B, T, E) * 2
    prior = F.softmax(torch.randn(B, T, E), dim=-1)
    commit = torch.tensor([1.0, 0.75, 0.56, 0.47, 0.39])
    resident = torch.zeros(E, dtype=torch.bool)
    resident[::3] = True
    target_logits = torch.randn(B, T, E)

    for shared in (False, True):
        for use_res in (False, True):
            g = MoESpeculativeGater(E, k, kmax, shared_across_batch=shared)
            out = g(draft_logits, draft_scores, commit, prior, resident if use_res else None)
            assert out.expert_mask.shape == (B, T, E) and out.expert_mask.dtype == torch.bool
            assert out.transfer_mask.shape == (B, E) and out.set_size.shape == (B,)
            assert (out.set_size <= torch.clamp(out.anchor_mask.sum(-1), min=kmax)).all() and (out.set_size >= k).all(), out.set_size
            assert (out.anchor_mask.sum(-1) == k).all() if not shared else (out.anchor_mask.sum(-1) >= k).all()
            eligible = out.expert_mask[:, 1:].any(dim=1)
            assert torch.equal(eligible & out.anchor_mask, out.anchor_mask), "anchors must stay eligible"
            assert (out.expert_mask[:, 0].sum(-1) == k).all(), "the anchor routes to exactly its natural top-k"
            assert not (out.pruned_mask & out.anchor_mask).any(), "anchors are never pruned"
            if use_res:
                assert not (out.pruned_mask & resident).any(), "only non-resident experts are pruned"
            if shared:
                assert (eligible == eligible[0]).all(), "one set for the whole batch"
            idx, wts = g.route_with_mask(target_logits, out.expert_mask, k)
            assert idx.shape == (B, T, k) and wts.shape == (B, T, k)
            assert out.expert_mask.gather(-1, idx).all(), "tokens may only use allowed experts"
            assert torch.allclose(wts.sum(-1), torch.ones(B, T)), "weights are a distribution over the k experts"

    # the hard cap at its minimum leaves only the anchor's experts
    g = MoESpeculativeGater(E, k, k)
    out = g(draft_logits, draft_scores, commit, prior)
    assert (out.set_size == k).all() and torch.equal(out.expert_mask[:, 1:].any(1), out.anchor_mask)
    # no demand beyond the anchor -> nothing added
    flat = torch.zeros(B, T, E)
    flat[:, :, :k] = 50.0
    out = MoESpeculativeGater(E, k, kmax)(draft_logits, flat, commit)
    assert (out.set_size == k).all(), out.set_size
    # commitment probabilities from traces
    p = MoESpeculativeGater.estimate_commit_probs(torch.tensor([0, 1, 1, 2, 4, 0, 3, 1]), 5)
    assert torch.allclose(p, torch.tensor([1.0, 0.75, 0.375, 0.25, 0.125])), p
    # half precision inputs and determinism
    g = MoESpeculativeGater(E, k, kmax)
    a = g(draft_logits.half(), draft_scores.half(), commit, prior.half())
    b = g(draft_logits.half(), draft_scores.half(), commit, prior.half())
    assert torch.equal(a.expert_mask, b.expert_mask)
    print("moe_speculative_gater: all checks passed;",
          f"example set sizes {g(draft_logits, draft_scores, commit, prior, resident).set_size.tolist()} of cap {kmax}")


if __name__ == "__main__":
    _self_test()
