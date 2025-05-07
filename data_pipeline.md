# Monte Carlo Data Pipeline

Libraries: VERL, MM-PRM, InterVL

## Generating Roll Outs
Take prompt and generate 4 solutions. (Solution A,B,C,D).

For each solution, allow model to generate maximum of 12 steps. Less is ok, if exceed then merge steps to fit into 12 only.

Assume Solution A,B,C,D has an average of 5.6 steps. 

For every 5.6 steps in solution A,B,C,D, we sample 16 continuations PER STEP.

For example, if solution A has 4 steps, then we sample 16 continuations for each step in solution A (S1, S2, S3, S4). We do this to generate MC values for S1, S2, S3, S4.

Tricky: For S1, the prefix is the question and S1 (fixed). We fix this "prefix" and sample 16 continuations for S1. The MC value of S1 is the proportion of correct "Final Answer" out of the 16 continuations, which we compare with the ground truth answer.

For S2, the prefix is the question and S1 and S2 (fixed). We fix this "prefix" and sample 16 continuations for S2. The MC value of S2 is the proportion of correct "Final Answer" out of the 16 continuations, which we compare with the ground truth answer.

We repeat this for S3 and S4. Which means for solution A, we would have sampled 64 continuations (16 * 4), BUT we only keep the 4 initial solutions (A, B, C, D), now with annotated step-by-step MC values.

Annotated Solutions A,B,C,D are then built into a multi-turn chat conversation where the conversation goes something like question -> S1 -> "+" if S1's MC value is greater than some threshold otherwise "-" -> S2 -> "+" if S2's MC value is greater than some threshold otherwise "-" -> S3 -> "+" if S3's MC value is greater than some threshold otherwise "-" -> S4 -> "+" if S4's MC value is greater than some threshold otherwise "-".

We then fine-tune a chat model using this conversation data using standard cross entropy loss (NTP) to predict "+" or "-" for each step. Constrained generation could be used to ensure the model only generates "+" or "-" for each step during inference time.

Saying this is Monte Carlo sampling is a bit of a misnomer, because we are not sampling from a probability distribution. We are sampling from a fixed set of 4 solutions, and then sampling 16 continuations for each step in the solution. In AlphaZero, they actually sample from a probability distribution, and they use a tree search to explore the space of possible solutions.

Why this sampling method?

The point of Monte Carlo is we want to systematically sample a tree of solutions. Unlike alpha zero, we cannot explore every single possible solution. So Monte Carlo tree search allows us to proxy that.

There are two variants of Monte Carlo Sampling:

1. "Vanilla MCTS (Math-Shepherd, MiPS): e.g. we sample 4 solutions for each image-question pair and split each of them into at most 12 steps. For each step, we sample 16 continuations and compute mci according to these continuations where mci = num(correct completions)/num(sampled completions).Annotate correctness of every step. (Linear, per-step annotation process)

2. Algorithmic MCTS: e.g. OmegaPRM which "algorithmically" generates rollouts using MCTS + binary search to efficiently exploring the space of possible paths using MCTS, focusing search on finding errors and building a representative tree from which training data is extracted. Basically prunes the search tree of solutions for more "high signal" data to train PRM on.

In variant 2, How Monte Carlo works at a high level is a mechanism called UCT which is inspired by AlphaZero's UCB. UCT stands for Upper Confidence Tree which comes from Alpha Confidence Bound. The point of UCB or UCT is to balance the tree's exploration and exploitation implementation. In OmegaPRM, a PUCT (Predictor + UCB, UCB for Tree Search) is used.

PUCT maximizes the likelihood of tree traversal paths that have the highest potential, which generates better rollouts compared to random rollouts.

For now, we will use variant 1, to focus on the "per-step" multimodal verification process.

## MC Value Labeling
After we have generated a tree of solutions, the next step is to label the goodness of each step in the tree, where each step is a single node in the tree, which is built by our MCTS.

The straightforward way of labeling each step with an MC value is to take the final outcome or the final answer of every leaf node, And at every note the MC value is the proportion of leaf nodes that reach the correct final answer. (Double check this, how do the scores propagate up the tree from N-1 depth to the current node's we are labelling?)

There is something called advantage-based mcvalue and label-based.

After labeling each node with an MC value, the next step is to train the process reward model with either a preference optimization method (DPO) or a next token prediction cross entropy loss.

