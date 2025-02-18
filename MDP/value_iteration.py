#This is based off the pseudo-code in slid 62 of Lecture 5 MDPs.  

states = ["s1", "s2", "s3"]
terminal_state = "s3"

actions = ["a1", "a2"]

gamma = 0.9
theta = 1e-6  # some threshold

transition_rewards = {
    ("s1", "a1"): ("s2", 2),
    ("s1", "a2"): ("s3", 5),
    ("s2", "a1"): ("s3", 1),
    ("s2", "a2"): ("s1", -1),
}

V = {s: 0 for s in states}

while True:
    delta = 0
    new_V = V.copy()

    for s in states:
        if s == terminal_state:
            continue

        max_value = float("-inf")
        for a in actions:
            if (s, a) in transition_rewards:
                s_prime, r = transition_rewards[(s, a)]
                value = r + gamma * V[s_prime] #transition probs are always one.
                max_value = max(max_value, value)

        new_V[s] = max_value
        delta = max(delta, abs(V[s] - new_V[s]))

    V = new_V

    if delta < theta:
        break


optimal_policy = {}
for s in states:
    if s == terminal_state:
        continue
    best_action = None
    best_value = float("-inf")

    for a in actions:
        if (s, a) in transition_rewards:
            s_prime, r = transition_rewards[(s, a)]
            value = r + gamma * V[s_prime]
            if value > best_value:
                best_value = value
                best_action = a

    optimal_policy[s] = best_action

print("Optimal Value Function:")
for state, value in V.items():
    print(f"V({state}) = {value:.4f}")

print("\nOptimal Policy:")
for state, action in optimal_policy.items():
    print(f"π*({state}) = {action}")
