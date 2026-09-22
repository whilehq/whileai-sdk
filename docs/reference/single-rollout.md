---
title: "One rollout per prompt, in symbols"
sidebarTitle: "Single rollout"
description: "The three single-rollout updates (FlashReinforce, SAO, BPCO) as the arithmetic update() performs, one block each, with the group update they replace for contrast and a table from each symbol to the field that carries it."
---

Every policy-gradient method in this library is one choice of a number
per token. A trainer's loss is

$$
\mathcal{L}(\theta) = -\sum_{i}\sum_{t} c_{i,t}\,\log \pi_\theta(a_{i,t} \mid h_{i,t}),
\qquad c_{i,t} \text{ held constant},
$$

so the gradient step is $\sum_{i,t} c_{i,t}\,\nabla_\theta \log \pi_\theta$.
`method.update(batch)` returns $c$ as `coefficients[i][t]`, and this page
says what $c$ is for each method. The plain-words version is
[lesson 9](/learn/train-on-production-traces); the numbers behind every
default are in `whileai/simulations/defaults.py`; the equation numbers are
the papers'.

## Notation

A batch holds $B$ trajectories. Trajectory $i$ has reward $R_i$ and
$T_i$ action tokens $a_{i,1} \ldots a_{i,T_i}$, each with history
$h_{i,t}$. Two policies see every token: $\mu$, the policy that wrote the
trace (a row's `behavior_logprobs`), and $\pi_\theta$, the one being
trained (`logprobs`). Their per-token ratio is

$$
\rho_{i,t} = \frac{\pi_\theta(a_{i,t} \mid h_{i,t})}{\mu(a_{i,t} \mid h_{i,t})}
= \exp\big(\texttt{logprobs}[t] - \texttt{behavior\_logprobs}[t]\big).
$$

A token the environment wrote (tool output, an observation) has
`action_mask` false: it is not one of the $T_i$, gets $c = 0$, and the
critic methods bootstrap over it. When `behavior_logprobs` is absent the
row is taken as on-policy, $\rho = 1$.

## The group update, for contrast

GRPO draws $k$ rollouts of the same prompt and centers each reward on its
siblings:

$$
A_i = \frac{R_i - \operatorname{mean}_{j \in \text{group}(i)} R_j}{\operatorname{std}_{j \in \text{group}(i)} R_j}.
$$

With one rollout per prompt the group is the rollout itself, $A_i \equiv 0$,
and the update is empty. The three methods below each replace that
group with something a single trajectory can supply.

## FlashReinforce

Baseline from the batch, a gate on drift, equal weight per trajectory.

$$
A_i = R_i - \frac{1}{B}\sum_{j=1}^{B} R_j
\qquad \text{(Eq. 5, no division by a standard deviation)}
$$

$$
d_{i,t} = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q},
\quad p = \mu(a_{i,t} \mid h_{i,t}),\; q = \pi_\theta(a_{i,t} \mid h_{i,t})
\qquad \text{(Eq. 6)}
$$

$$
\bar D_i = \frac{1}{T_i}\sum_{t=1}^{T_i} d_{i,t},
\qquad
m_i = \mathbf{1}\!\left[\bar D_i \le \delta\right]
\qquad \text{(Eq. 7, 8)}
$$

$$
c_{i,t} = \frac{m_i\,A_i\,\rho_{i,t}}{T_i\,B}
\qquad \text{(the gradient of Eq. 9 with } \rho, A, m \text{ fixed; Eq. 10)}
$$

$d_{i,t}$ is the KL between two Bernoullis, "this token versus everything
else," so it needs only the one probability the sampler already stored.
A trajectory over the gate drops out whole; the ratio is never clipped,
only kept finite. Defaults: $\delta$ = `trust` = 0.003, a lag of
`off_policy_steps` = 8 updates.

## SAO

A critic in place of the group, credit passed back over tool output, and
a band that masks a drifted token instead of clipping it.

Over the $L$ action tokens of one trajectory, in order, with $r_k = R$ on
the last one and $0$ before it, $V$ the critic's value at that token and
$V = 0$ past the end:

$$
\delta_k = r_k + \gamma\,V(a_{k+1}) - V(a_k),
\qquad
A_k = \delta_k + \gamma\lambda\,A_{k+1},
\qquad
\lambda = 1 - \frac{1}{\alpha L}
\qquad \text{(Eq. 4, 5)}
$$

$$
f(\rho) = \begin{cases} \rho & 1 - \varepsilon_{\text{low}} < \rho < 1 + \varepsilon_{\text{high}} \\ 0 & \text{otherwise} \end{cases}
\qquad \text{(Eq. 2, 3)}
$$

$$
c_{i,t} = \frac{f(\rho_{i,t})\,A_{i,t}}{N},
\qquad N = \text{action tokens in the batch}
\qquad \text{(Eq. 1)}
$$

The bootstrap $V(a_{k+1})$ is the next *action* token, so an observation
between two actions is skipped, not scored. $\lambda$ grows with length so
the terminal reward reaches the first token with weight
$\lambda^{L-1} \approx e^{-1/\alpha}$ whatever $L$ is. No normalization:
the critic is the baseline. The critic trains toward the Monte Carlo
return, $R$ at every action token, with a squared-error loss, `critic_steps`
= 2 updates per policy update after `critic_warmup` = 10 critic-only
steps. Defaults: $\alpha$ = `gae_alpha` = 1.5, $\gamma$ = 1, the band
`ratio` = $(1 - \varepsilon_{\text{low}},\, 1 + \varepsilon_{\text{high}})$ = (0.7, 6.0).

## BPCO

A critic that cannot leave the reward's range, trained on the reward the
rollout earned, an advantage left at its natural scale, and a clip on
probability rather than on ratio.

$$
V = R_{\min} + (R_{\max} - R_{\min})\left(\tfrac{1}{2} + \tfrac{1}{\pi}\arctan z\right)
\qquad \text{(Eq. 9; } z \text{ the raw head output; } \texttt{bound(z)}\text{)}
$$

$$
\hat V_t = R \quad \text{for every token} \qquad (\lambda_V = 1,\ \gamma = 1;\ \text{Eq. 11})
$$

$$
\delta_t = r_t + V(s_{t+1}) - V(s_t),
\qquad
A_t = \delta_t + \lambda_\pi A_{t+1},
\qquad
\lambda_\pi = 1 - \frac{1}{\alpha L}
\qquad \text{(Eq. 3, 4, 14; not normalized)}
$$

$$
\text{surrogate}_t = \min\!\Big(\rho_t A_t,\ \operatorname{clip}\!\big(\rho_t,\ 1 - \tfrac{\varepsilon}{\mu_t},\ 1 + \tfrac{\varepsilon}{\mu_t}\big) A_t\Big),
\qquad \mu_t = \mu(y_t \mid s_t)
\qquad \text{(Eq. 2, DPPO)}
$$

$$
c_{i,t} = \begin{cases} \rho_{i,t} A_{i,t} & \text{the unclipped branch is the minimum} \\ 0 & \text{the clipped branch is} \end{cases}
$$

The clip range is $\varepsilon / \mu_t$: a token the old policy gave
probability 0.05 may move its ratio four times further than one it gave
0.2, because the same ratio change moves less probability mass. The loss
sums a sequence's tokens and averages over sequences, so no length weight
enters $c$. The critic trains alone for `critic_warmup` = 15 updates first.
Defaults: $\varepsilon$ = `clip` = 0.2, $\alpha$ = `gae_alpha` = 0.4,
`reward_range` = $(R_{\min}, R_{\max})$ = (0, 1).

## From symbol to field

| Symbol | `Update` field | Notes |
|---|---|---|
| $c_{i,t}$ | `coefficients[i][t]` | the number a trainer multiplies $\nabla \log \pi_\theta$ by; 0 on a masked token |
| $A_i$ or $A_{i,t}$ | `advantages[i][t]` | before any ratio, gate, band or clip; constant along a FlashReinforce trajectory |
| $m_i$, or "some token inside the band" | `admitted[i]` | a trajectory with no gradient left |
| $\hat V_t$ | `value_targets[i][t]` | `None` for FlashReinforce, which has no critic |
| $\bar D_i$, $\rho$, masked or clipped share | `stats` | what a run page plots per step |
| what was dropped and why | `notes` | the sentence a person reads first |

Every default above is a field on the object, refused on construction
when it is out of range, and named with its source in `defaults.py`. What
each method did on a GPU, against its own ablation, is the table in
[lesson 9](/learn/train-on-production-traces#what-is-proven-and-what-is-not)
and the three recipes under [papers](/recipes/papers).
