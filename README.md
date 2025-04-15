Current Algorithm: Tabular Q-Learning
Mechanism:

Uses a Q-table to store state-action values

Updates values using the Bellman equation:

python
Copy
Q[s,a] = (1-α)Q[s,a] + α(r + γ*max(Q[s']))
Optimizations Added:

ε-greedy exploration

Reward shaping (distance-based rewards)

State space simplification (grid + food direction + danger detection)
