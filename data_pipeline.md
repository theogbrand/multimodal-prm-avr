# Monte Carlo Data Pipeline

Libraries: VERL, MM-PRM, InterVL

## Generating Roll Outs

 We use a Monte Carlo tree search algorithm to generate synthetic data 

The point of Monte Carlo is we want to systematically sample a tree of solutions. Unlike alpha zero, we cannot explore every single possible solution. So Monte Carlo tree search allows us to proxy that.

How Monte Carlo works at a high level is a mechanism called UCT which is inspired by AlphaZero's UCB. UCT stands for Upper Confidence Tree which comes from Alpha Confidence Bound. The point of UCB or UCT is to balance the tree's exploration and exploitation implementation.

UCT uses something called best first search which essentially maximizes the likelihood of tree traversal paths that have the highest potential.

## MC Value Labeling
After we have generated a tree of solutions, the next step is to label the goodness of each step in the tree, where each step is a single node in the tree, which is built by our MCTS.

The straightforward way of labeling each step with an MC value is to take the final outcome or the final answer of every leaf node, And at every note the MC value is the proportion of leaf nodes that reach the correct final answer. (Double check this, how do the scores propagate up the tree from N-1 depth to the current node's we are labelling?)

There is something called advantage-based mcvalue and label-based.

After labeling each node with an MC value, the next step is to train the process reward model with either a preference optimization method (DPO) or a next token prediction cross entropy loss.

