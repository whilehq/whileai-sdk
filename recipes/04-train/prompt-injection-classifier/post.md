# A 22M injection classifier, and what its own data taught it

Open models ship with no defence against a prompt injection: an instruction
planted in a web page, a tool result or an email that the agent reads as if
the user had said it. The published guards are 184M parameters and were
trained on direct attacks in the user turn. We wanted the cheap layer: a
classifier small enough to run on every chunk an agent reads, on a CPU, in
front of the model, and trained on the channel that matters, the indirect
one.

The whole recipe is one idea and one trap. The idea: every attack string is
public (InjecAgent, AgentDojo, BIPIA, Gandalf), so plant it by program into
a carrier and the label is known by construction. The trap: the carrier is
then the recipe's own, and a 22M encoder learns the carrier in nineteen
seconds. Round 1 scored AUROC 1.00 on every planted slice and flagged every
sentence in NotInject, the benign trigger-word set. It had learned the
generator.

Three things fixed it, and each one is a rule from the RLHF book (Lambert 2025,
rlhfbook.com) applied to a classifier. Matched twins (round 3): every planted row gets a sibling with
the same carrier, position and framing and a harmless insert of the same
length, so the only thing that separates the pair is whether the insert is
an instruction to the model. That is a preference pair with the confound removed (the *Direct Alignment*
chapter's shape, length-matched as the *Preference Data* chapter asks), and it
took NotInject from 0 to 97 points. Channel coverage over volume (rounds 4 and 5): 16,000
rows from a permissive "prompt injection" set that is mostly role-play
jailbreaks moved three headline slices down; 3,500 rows of the two missing
channels (a document pasted under a user ask, and agent traffic across six
businesses) moved three up. And carriers from a model, not a template (round
7, the *Synthetic Data and Distillation* chapter's diverse-teacher rule, with
real human text mixed in): `wai.simulate` with a model as situation writer, as the user and, through
`execute=`, as the world, so each tool call comes back as the document that
tool would return. The bag-of-words probe on the indirect slices fell from
0.82 AUROC to 0.63 when the carriers changed, which is the shortcut leaving
the data.

Two things did not work and are on the page as results. A pairwise hinge on
the twins (round 6) bought nothing, because by then 1.4% of training pairs
were in the uncertain band and a margin on pairs the model already separates
moves the logit scale, not the decision. Hard-negative mining (the *Rejection
Sampling* chapter's keep-what-is-hard, with its random-selection control)
found the same saturated pool.

Where it stands: on the business held out of training the round-8 model scores 73 points to the baseline's 49 with 0 to 2 false positives in 211 documents; on simulated tool results 93 to 42; on NotInject 94 to 57; on the hard held-out-family test 51 to 48 (60 for round 5). It loses to the 184M baseline on deepset,
a direct-attack set with German rows, on every round, and says so. The 5 ms
latency target at 512 tokens is missed on an ARM laptop (44 ms; 7.5 ms per
128-token window), so the serving story is a window with early exit, not a
single pass.

The recipe: `recipes/04-train/prompt-injection-classifier`. The climb, with
every round's five-line note and the shortcut probes, is on the platform
under `prompt-injection-classifier`.
