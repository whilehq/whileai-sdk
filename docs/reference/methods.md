---
title: "The methods, in symbols"
sidebarTitle: "Methods"
description: "Every training method the library names, as one block of arithmetic each: the hosted four (SFT, GRPO, DPO, RM), the distillation pair (OPD, OPSD), the staleness corrections (Async), the group baselines prime-rl runs, adaptive sampling (Reinforce-Ada), and the three single-rollout updates, with a table from each symbol to the field that carries it."
---

Every method here is one loss over the tokens a model wrote. For the
policy-gradient family the loss is

$$
\mathcal{L}(\theta) = -\sum_{i}\sum_{t} c_{i,t}\,\log \pi_\theta(a_{i,t} \mid h_{i,t}),
\qquad c_{i,t} \text{ held constant},
$$

so the gradient step is $\sum_{i,t} c_{i,t}\,\nabla_\theta \log \pi_\theta$, and
a method is a choice of the per-token coefficient $c$. The others (SFT,
DPO, a reward model, distillation) are a different loss, written out the
same way. The plain-words course is [Learn](/learn); the numbers behind
every default are in `whileai/simulations/defaults.py` with their
sources; equation numbers are the papers'. Each method's own page says
where it runs: the hosted trainer (`wai.train`), prime-rl
(`wai.prime_rl_config`), or your own loop (`method.update`).

## Notation

A batch holds $B$ trajectories. Trajectory $i$ answers prompt $x_i$ with
reward $R_i$ and $T_i$ action tokens $a_{i,1} \ldots a_{i,T_i}$, each with
history $h_{i,t}$ (the prompt and the tokens before it). Two policies can
see every token: $\mu$, the policy that wrote it (a row's
`behavior_logprobs`; the rollout engine, or last week's served model), and
$\pi_\theta$, the one being trained (`logprobs`). Their per-token ratio is

$$
\rho_{i,t} = \frac{\pi_\theta(a_{i,t} \mid h_{i,t})}{\mu(a_{i,t} \mid h_{i,t})}
= \exp\big(\texttt{logprobs}[t] - \texttt{behavior\_logprobs}[t]\big),
$$

and $\pi_{\text{ref}}$ is a frozen reference (the starting weights) where a
method keeps one. A token the environment wrote (tool output, an
observation) has `action_mask` false: it is not one of the $T_i$, gets
$c = 0$, and the critic methods bootstrap over it. When `behavior_logprobs`
is absent the row is taken as on-policy, $\rho = 1$.

## The hosted four: `wai.train(method=)`

**SFT.** Maximum likelihood on the rows as pushed, over the assistant
tokens the loss mask keeps (the trainer does not filter on reward, so
push `scored.passes()`):

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_i \sum_{t=1}^{T_i} \log \pi_\theta(a_{i,t} \mid h_{i,t}).
$$

**GRPO** (Shao et al. 2024, arXiv:2402.03300). $k$ = `generations`
rollouts of the same prompt form a group; the advantage is the reward
centered on its siblings, and the coefficient is PPO's clipped surrogate
with a KL to the reference:

$$
A_i = \frac{R_i - \operatorname{mean}_{j \in \text{group}(i)} R_j}{\operatorname{std}_{j \in \text{group}(i)} R_j},
\qquad
\mathcal{L}_{\text{GRPO}} = -\sum_i w_i \sum_{t} \min\!\big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\, A_i\big) + \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}).
$$

A group whose rollouts all pass or all fail has $A_i = 0$ for every
member, which is why `select(mode="rl")` drops unanimous groups before
the GPU is spent. `loss_type` picks $w_i$, the length weight:
`grpo` averages each sequence's tokens then averages sequences
($w_i = 1/T_i$); `bnpo` pools every token in the batch ($w_i = 1/\sum_j T_j$);
`dr_grpo` (Liu et al. 2025, arXiv:2503.20783) drops the standard deviation
from $A_i$ and uses one constant length ($w_i = 1/L_{\max}$), removing the
bias toward long wrong answers. `truncated="mask"` gives a reply the token
cap cut $c = 0$; `"zero"` scores it $R = 0$ instead. Defaults: $k$ = 8,
$\varepsilon$ = 0.2, $\beta$ = 0 (DAPO, Dr. GRPO and CISPO all run without
the KL term).

**DPO** (Rafailov et al. 2023, arXiv:2305.18290). A pass $y_w$ and a fail
$y_l$ of the same prompt, length matched; no sampling, no reward model:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right),
\qquad \beta = 0.1.
$$

**RM.** The same pairs train a scalar head $r_\phi$ under the
Bradley-Terry model; `reward_model(run)` is then a judge:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\log \sigma\!\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big).
$$

## Distillation: `wai.OPD`, `wai.OPSD`

**OPD**, on-policy distillation (Agarwal et al. 2023, arXiv:2306.13649;
Thinking Machines 2025). The student samples its own reply; a frozen
teacher $\pi_T$ scores every token of it. The per-token loss is the
reverse KL from the student to the teacher over the teacher's top-$k$
tokens (Li et al. 2026, arXiv:2604.13016; Fu et al. 2026, arXiv:2603.25562),
and its sampled-token form is a policy gradient whose advantage is the
log-probability gap:

$$
\mathcal{L}_{\text{OPD}}(\theta) = \sum_i \sum_t \sum_{v \in \operatorname{top}_k \pi_T(\cdot \mid h_{i,t})} \pi_\theta(v \mid h_{i,t}) \log \frac{\pi_\theta(v \mid h_{i,t})}{\pi_T(v \mid h_{i,t})},
\qquad
c_{i,t} = \log \pi_T(a_{i,t} \mid h_{i,t}) - \log \pi_\theta(a_{i,t} \mid h_{i,t}).
$$

No reward, no group, no discount: a token the teacher liked more than the
student did is pushed up by the gap. `divergence` offers `forward_kl` and
`jsd` for a trainer that has them. Defaults: $k$ = `top_k` = 32,
`samples` = 4, temperature 1.0 on both sides so the teacher scores the
distribution the student drew from.

**OPSD**, on-policy self-distillation (SDFT, Shenfeld et al. 2026,
arXiv:2601.19897; SDPO, Hübotter et al. 2026, arXiv:2601.20802). The same
loss, with the teacher the student's own weights shown a hint $p$ the
student never sees (a passing demonstration, the reference answer, a hint,
or a rollout plus the environment's feedback, `privileged`):

$$
\pi_T(\cdot \mid h_{i,t}) = \pi_{\bar\theta}(\cdot \mid h_{i,t} \oplus p_i),
\qquad
\bar\theta \leftarrow (1-\alpha)\,\bar\theta + \alpha\,\theta \quad (\texttt{anchor="ema:0.01"}).
$$

The anchor is what keeps the teacher from chasing the student:
`"ema:0.01"` is the moving average above, `"initial"` freezes the starting
weights (Zhao et al. 2026, arXiv:2601.18734, with `forward_kl`), `"live"`
is the unanchored student prime-rl runs. `samples` = 1: there is no group
to form.

## Staleness: `wai.Async`

Any method above, trained on rollouts that lag the policy by at most
`off_policy_steps` = $s$ optimizer steps, with one per-token correction
for the sampler/trainer gap that exists even at zero lag (Yao et al. 2025):

$$
\texttt{ipo}:\ c_{i,t} \leftarrow c_{i,t}\,\mathbf{1}\big[\,|\pi_\theta(a_{i,t}\mid h_{i,t}) - \mu(a_{i,t}\mid h_{i,t})| \le \varepsilon\,\big],\ \varepsilon = 0.3
$$

$$
\texttt{icepop}:\ c_{i,t} \leftarrow c_{i,t}\,\mathbf{1}\big[\,\rho_{\text{low}} \le \rho_{i,t} \le \rho_{\text{high}}\,\big],\ (\rho_{\text{low}}, \rho_{\text{high}}) = (0.5, 5.0)
$$

$$
\texttt{tis}:\ c_{i,t} \leftarrow c_{i,t}\,\min(\rho_{i,t}, \rho_{\max}),\ \rho_{\max} = 2.0
$$

`ipo` masks on probability moved (prime-rl's default), `icepop` masks
outside a ratio band (Ring-1T, arXiv:2510.18855), `tis` caps the ratio
(verl). Default $s$ = 8 (ScaleRL, arXiv:2510.13786; one step is free,
Noukhovitch et al. 2024, arXiv:2410.18252).

## The group baselines prime-rl runs: `"grpo"`, `"max_rl"`, `"rae"`

Three string methods `prime_rl_config` writes, differing only in $A_i$:

$$
\texttt{grpo}:\ A_i = R_i - \operatorname{mean}_{\text{group}} R
\qquad
\texttt{max\_rl}:\ A_i = \frac{R_i - \operatorname{mean}_{\text{group}} R}{\operatorname{mean}_{\text{group}} R}
\qquad
\texttt{rae}:\ A_i = R_i - b_{\text{agent}(i)},\ \ b \leftarrow 0.95\, b + 0.05\, R_i
$$

`max_rl` (arXiv:2602.02710) divides by the group mean rather than its
standard deviation, which makes the gradient unbiased for the
maximum-likelihood objective and weights a rarely solved prompt by about
$1/p$; a group whose mean is 0 carries nothing. `rae` (SPIRAL,
arXiv:2506.24119) is a per-agent running average, the one baseline that
stands at one rollout per prompt, and on a single-agent taskset it is
REINFORCE against that average.

## One rollout per prompt: `wai.FlashReinforce`, `wai.SAO`, `wai.BPCO`

The shape a production trace arrives in. With one rollout per prompt every
group above is the rollout itself, $A_i \equiv 0$, and the update is
empty. Each of these three replaces the group with something a single
trajectory can supply, and each one's `update(batch)` returns the $c$
below as `coefficients`.

### FlashReinforce

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

### SAO

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

### BPCO

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

## Adaptive sampling: `wai.methods.ReinforceAda`

Reinforce-Ada (Xiong et al. 2025, arXiv:2510.04996) changes which
rollouts a group-relative update sees, not the loss. With a binary reward
and a group of $k$, a prompt the policy solves with probability $p$ comes
back unanimous, and so with $A_i = 0$ for every rollout, with probability

$$
P_{\text{flat}}(p) = p^{k} + (1-p)^{k},
\qquad P_{\text{flat}}(0.1) = 0.66 \text{ at } k = 4.
$$

The paper's reason to care is the objective. Maximizing
$J_f = \mathbb{E}_x\,[f(p_\theta(x))]$ for a concave $f$ gives

$$
\nabla_\theta J_f = \mathbb{E}_x\big[f'(p_\theta(x))\,\nabla_\theta p_\theta(x)\big],
\qquad f = \log \ \Rightarrow\ f'(p) = \frac{1}{p},
$$

so under $\log p$ the prompts the policy rarely solves weigh the most,
the same $1/p$ that `max_rl` puts in the advantage. Reinforce-Ada puts it
in the sampling budget instead. Each round draws $M$ = `round_size`
rollouts for every prompt still active, and a prompt retires once its
pool holds $\lfloor k/2 \rfloor$ right and $\lceil k/2 \rceil$ wrong
(`exit="balanced"`) or one right (`exit="positive"`), for at most
$N_{\max}$ = `round_size` $\times$ `max_rounds` draws. From the $N_x$
drawn for prompt $x$ it keeps $k$ = `keep`, balanced where the pool
allows, and sets

$$
\hat p_x = \frac{1}{N_x}\sum_{j=1}^{N_x} \mathbb{1}[r_j > \tau],
\qquad
A_i = r_i - \hat p_x \quad (i \text{ among the } k \text{ kept}),
$$

with $\tau$ = `threshold`. The baseline is the whole pool's pass rate,
not the kept group's mean: the kept group is balanced on purpose, so its
mean is $1/2$ whatever the prompt's difficulty. There is no division by a
standard deviation. The coefficient is then GRPO's, and the backward pass
is the same $k$ rollouts per prompt it always was; the cost is generation,
$\mathbb{E}[N_x]$ draws per prompt instead of $k$.

```python
import random

import whileai as wai

rates = [0.05, 0.3, 0.6, 0.95]  # how often the model solves each prompt
rng = random.Random(0)


def draw(prompts: list[int], k: int) -> list[list[float]]:
    """k rewards per prompt; in a real run, k rollouts graded by your verifier."""
    return [[float(rng.random() < rates[p]) for _ in range(k)] for p in prompts]


result = wai.methods.ReinforceAda()(draw, list(range(len(rates))))
print(result)  # draws per prompt, share retired, share with no gradient vs GRPO at 4
```

In TRL, `wai.methods.ReinforceAda().trainer(GRPOTrainer)` is a
`GRPOTrainer` whose generation step is this sampler (trl 0.19, with
`num_generations = keep`, `num_iterations = 1`, `beta = 0`). The
replication is
[`recipes/papers/reinforce-ada`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/reinforce-ada):
on GSM8K with Qwen2.5-1.5B it cut the prompts with no gradient from
0.52-0.62 to 0.25-0.33 of each batch at 2.5 times the GPU minutes, and
moved pass@1 by +0.047 [-0.047, +0.140] across two training seeds per
arm, flat. The prompts it could not rescue were ones the model always
solves, where 32 draws find no wrong answer; `exit="positive"` does not
spend draws on those. Defaults: `keep` = 4, `round_size` = 8,
`max_rounds` = 4, `threshold` = 0.7, the authors' own, each named in
`whileai/reinforce_ada.py`.

## Rollouts that refine each other: `wai.methods.Swarm`

Particle swarm optimization over rollouts (Kennedy and Eberhart 1995).
Round 0 is `particles` fresh samples. Each later round a particle is shown
its own best attempt with the fitness's feedback and, by `topology`, a
neighbour's best, and writes the next attempt: `solo` shows no neighbour,
`ring` the better of two ring neighbours, `star` the best in the swarm.
The model is the velocity update. It is a rollout rule, not a trainer: it
decides which samples get drawn, not how the policy moves.

```python
import re

import whileai as wai


def fitness(text: str) -> dict:
    m = re.search(r"print\((\d+)\)", text)
    got = int(m.group(1)) if m else 0
    return {"fitness": 1 - abs(42 - got) / 42, "correct": got == 42, "feedback": f"printed {got}"}


def model(messages: list[dict], seed: int) -> str:
    """Any (messages, seed) -> text call; or pass a backend such as
    wai.Endpoint("Qwen/Qwen3-4B", url="http://localhost:8000/v1")."""
    return "print(42)" if "teammate" in messages[-1]["content"] else "print(41)"


result = wai.methods.Swarm(model, topology="ring")("Print 42.", fitness)
print(result)
print(wai.methods.Swarm.calibration(result.samples))
```

It tied plain resampling at the same budget on every task family in
[`recipes/01-simulate/swarm-rescue`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/01-simulate/swarm-rescue):
partial credit on tests predicted a pass zero times below three quarters
of the tests, a model judge ranked passes at chance, and where the fitness
was a real hill the swarm learned the shown tests instead of the task.
`Swarm.calibration(rows)` prints P(correct | fitness bucket) over any
graded rows; a fitness with no correct attempt below full credit is a
cliff, and a swarm has nothing to climb on it. Defaults: `particles` = 8,
`rounds` = 3, `temperature` = 1.0, `max_tokens` = 2048, each named and
justified in `whileai/swarm.py`.

## From symbol to field

For the three single-rollout methods, `update(batch)` returns an `Update`:

| Symbol | `Update` field | Notes |
|---|---|---|
| $c_{i,t}$ | `coefficients[i][t]` | the number a trainer multiplies $\nabla \log \pi_\theta$ by; 0 on a masked token |
| $A_i$ or $A_{i,t}$ | `advantages[i][t]` | before any ratio, gate, band or clip; constant along a FlashReinforce trajectory |
| $m_i$, or "some token inside the band" | `admitted[i]` | a trajectory with no gradient left |
| $\hat V_t$ | `value_targets[i][t]` | `None` for FlashReinforce, which has no critic |
| $\bar D_i$, $\rho$, masked or clipped share | `stats` | what a run page plots per step |
| what was dropped and why | `notes` | the sentence a person reads first |

For the rest, the symbols are the knobs: $k$ is `generations`, $\beta$ is
`beta`, $\varepsilon$ is the trainer's clip, $w_i$ is `loss_type`, $s$ is
`off_policy_steps`, $\alpha$ is the rate in `anchor`, and every one is a
named constant in `defaults.py` with the paper it came from. What the
single-rollout three did on a GPU, against their own ablations, is the
table in [lesson 9](/learn/train-on-production-traces#what-is-proven-and-what-is-not)
and the three recipes under [papers](/recipes/papers).
