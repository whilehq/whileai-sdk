"""CISPO for prime-rl's `[trainer.loss] type = "custom"` hook.

prime-rl (pin 9ef21a64f) ships two rl losses: IPO (masks tokens whose probability moved
more than eps, default) and IcePop (masks tokens outside a fixed ratio band). Neither is
CISPO, which ScaleRL (Khatri et al. 2025, arXiv:2510.13786, sec. 3.2 and 4) picks for its
final recipe: a truncated importance-sampling REINFORCE loss that keeps a gradient on every
token (no token is ever dropped), and is "markedly more robust" to its one knob than
DAPO-style clipping (App. A.17.4). Eq. 4:

    J = E[ sg(min(rho_t, eps_max)) * A * log pi_train(y_t) ]

Here rho_t = pi_trainer / pi_inference: prime-rl takes one optimizer step per batch, so the
ratio only carries trainer/vLLM numerical mismatch plus async staleness (up to
`max_off_policy_steps`). Token aggregation is prime-rl's (sum / global loss-token count).

Imported only by the trainer process; the env package's __init__ does not import it.
"""

from __future__ import annotations

import torch
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    compute_importance_ratio_and_mismatch_kl,
)


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].float().sum() / torch.clamp_min(mask.sum(), 1)


def cispo_loss(inputs: LossInputs, eps_max: float = 4.0) -> LossOutputs:
    _log_ratio, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        inputs.trainer_logprobs, inputs.inference_logprobs
    )
    mask = inputs.loss_mask
    ratio_sg = ratio.detach()
    # Masked (non-loss) positions can carry garbage ratios; zero them before they touch the loss.
    weight = torch.where(mask, torch.clamp(ratio_sg, max=eps_max), torch.zeros_like(ratio_sg))
    per_token = -(weight * inputs.advantages * inputs.trainer_logprobs)
    if inputs.loss_weights is not None:
        per_token = per_token * inputs.loss_weights
    metrics = {
        "cispo/truncated": _mean(ratio_sg > eps_max, mask),
        "cispo/is_weight": _mean(weight, mask),
        "cispo/mismatch_kl": _mean(mismatch_kl.detach(), mask),
    }
    return LossOutputs(loss=per_token.sum(), metrics=metrics)
