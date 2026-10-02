A 1.5B model can learn to manage its own context.

We reproduced Context Language Models (arXiv:2609.37725) with whileai. The model keeps its context as a file and rewrites it after every chunk of a log it never sees whole. Qwen2.5-1.5B went from 0.07 to 0.97 on held-out logs, on all four seeds. The paper's efficiency term added +0.02 and cut tokens 15%.

Our own guard against a shortcut lost to the paper's rule, and the write-up says so. 481 GPU min, $32, one H100 at a time.

wai.methods.ContextFile().trainer(GRPOTrainer)

Recipe: https://github.com/whilehq/whileai-sdk/tree/main/recipes/papers/context-lm
Guide: https://docs.withwhile.com/context-lm
Paper: https://arxiv.org/abs/2609.37725
