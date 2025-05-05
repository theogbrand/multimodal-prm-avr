# Visual Step By Step (VSBS)
related works: OmegaPRM, Math-Shepherd, MiPS, LLM-TS, VisualPRM, AgentPRM, GenPRM

Use Vanilla MCTS recipe for automated visual step by step reasoning generation. VisualPRM has reference implementation of doing this. 

Explore AgentPRM for label-free visual tuning, "direct" learning of step by step labels.

# Background
Our objective is to apply the AlphaZero method to LLMs, which requires modifications due to certain properties - LLMs have large finite states, while games have finite states. 

In AlphaZero, MCTS is used to plan according to a learned prior. A separate value network predicts the outcome of a position to choose paths. 

To apply this to LLMs, we use a language model as the prior (generator/policty model), and a separate language model as the value network (PRM/ORM).

# AlphaZero-like Tree-Search (Feng et al., 2023): Why Reward Models are needed for LLMs
Unlike Tree of Thought’s prompt-based heuristics, Tree Search LLM (TS-LLM) trains its own value network to score partial solutions (akin to a learned verifier). It uses this value model during search and also in a training loop.

During inference, TS-LLM runs deep tree search (e.g. MCTS) guided by the value model and a policy (LLM) to expand solution branches, returns Best-First Search (BFS) results.

Ultimately, the reward model will "guide" the policy model to generate a better solution which outperforms majority voting (self-consistency) and beam search baselines.

Conceptually, AlphaZero’s MCTS (planning) and LLM reward models occupy similar roles but operate differently.  In AlphaZero, search explicitly explores alternative future paths and uses the value function (analogous to an ORM) to evaluate outcomes, backing up these values to guide move selection - This assumes all possible solutions are known.

In LLMs, no explicit search is usually performed, knowing all possible solutions is not feasible. Instead, models generate or sample reasoning steps according to their policy (the language model itself), and reward models score those outputs.  However, a PRM can be seen as analogous to AlphaZero’s intermediate value estimates: it provides a quality score for any partial solution.  Indeed, Zhang et al. propose a variant of MCTS (called MCTS*) that uses a PRM as its value function, so that the search can evaluate and backpropagate rewards at intermediate nodes without completing full trajectories. ("Online RL")

ReST-MCTS* (Zhang et al., NeurIPS 2024) integrates a PRM with a specialized MCTS algorithm: it runs a tree search over chains of thought, using a trained PRM to score each intermediate step, and then backs up these “quality” values through the search tree.  This approach simultaneously self-trains the LLM policy and the PRM: given known correct answers, it uses MCTS to infer which steps in a reasoning path are truly helpful, and uses those inferred rewards to refine the PRM and select high-quality reasoning trace. ("Online RL")

And frameworks like PRIME (Qi et al., 2025) explicitly use policy rollouts with outcome checks to update a PRM online, without requiring manual step labels – essentially imitating how MCTS would generate and score trajectories.

We could use LLMs as verifiers, but they are limited for reasoning, hence we seek to train better verifiers.

## Training Reward Model (Offline RL, phase 1 of our project)
In AlphaZero, an outcome-based value network is trained on final game results. 

In LLMs, an ORM explicitly learns that outcome (e.g. answering correctly) by treating the entire solution as its “game result.”  Likewise, PRMs approximate the Q-value of partial solutions: they **estimate the “potential” of a reasoning prefix** to lead to a correct answer.

PRMs are essentially approximating the "Q-value" of a partial solution. They estimate the 'potential' of a reasoning prefix to lead to a correct answer.

PRMs trained by human annotations works but is costly (like PRM800K), so we seek automated methods. The two well-known methods are highlighted in MATH-Shepherd and OmegaPRM. 

The well-known method to training PRMs for math is given a math problem with a golden answer and a step-by-step solution, to achieve the label of a specific step, we utilize a fine-tuned LLM to decode multiple subsequent reasoning paths from this step. We further validate whether the decoded final answer matches with the golden answer. If a reasoning step can deduce more correct answers than another, it would be assigned a higher correctness score.

### Outline of Data Synthesis for generating data to train PRMs
1. Sample 

2. Label Step by Step


# Phase 2:  extended training the verifier by applying it to fine-tune the reasoner through reinforcement learning