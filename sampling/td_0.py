import numpy as np

# Define the episodes
episodes = [
    ["A", 0, "C", 0, "A", 0, "C", 1],
    ["A", 0, "C", 1],
    ["B", 0, "C", 0],
    ["B", 0, "C", 0, "A", 0, "C", 0],
    ["C", 1],
    ["C", 1],
]

# Parameters
states = ["A", "B", "C"]
alpha = 0.1  # Step size
gamma = 1  # Discount factor

# Initialize value function
V = {state: 0 for state in states}


# Function to compute the TD(0) updates for a single batch
def compute_batch_update(episodes, V, alpha, gamma):
    updates = {state: 0 for state in states}  # Store the total updates per state
    visit_count = {state: 0 for state in states}  # Count visits per state

    # Process each episode
    for episode in episodes:
        for i in range(0, len(episode) - 1, 2):  # Iterate over state, reward pairs
            S = episode[i]
            R = episode[i + 1]
            if i + 2 < len(episode):
                S_next = episode[i + 2]
                updates[S] += R + gamma * V[S_next] - V[S]
            else:
                updates[S] += R - V[S]

            visit_count[S] += 1

    # Apply the batch update
    for state in states:
        if visit_count[state] > 0:  # Update only visited states
            V[state] += alpha * (updates[state] / visit_count[state])


def main():
    # Perform multiple batch updates
    num_batches = 1000  # Number of batches to run
    for batch in range(num_batches):
        compute_batch_update(episodes, V, alpha, gamma)
        print(f"After batch {batch + 1}: {V}")


if __name__ == "__main__":
    main()
