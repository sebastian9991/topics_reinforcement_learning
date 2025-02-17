#This is based off of psuedo-code is sourced from lecture MDP slide 39.
import numpy as np

states = ["s1", "s2", "s3"]
terminal_state = "s3"

gamma = 0.9
theta = 1e-6  # Some threshold

policy = {"s1": "a1", "s2": "a1"}

transition_rewards = {
    ("s1", "a1"): ("s2", 2),
    ("s1", "a2"): ("s3", 5),
    ("s2", "a1"): ("s3", 1),
    ("s2", "a2"): ("s1", -1),
}

#initialize V(s) = 0 for all states
V = {s: 0 for s in states}

while True:
    delta = 0
    for s in states:
        if s == terminal_state:
            continue

        v = V[s]
        a = policy[s]
        s_prime, r = transition_rewards[(s, a)]

        #update
        V[s] = r + gamma * V[s_prime] #Transition probability is always one here.

        delta = max(delta, abs(v - V[s]))

    if delta < theta:
        break

for state, value in V.items():
    print(f"V({state}) = {value:.4f}")
