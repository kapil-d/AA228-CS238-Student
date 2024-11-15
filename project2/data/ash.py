import numpy as np
import pandas as pd


class BatchQLearning:
    def __init__(self, n_states, n_actions, gamma, alpha):
        self.Q=np.zeros([n_states, n_actions]) #initialize all Q(s,a) to be 0
        self.gamma = gamma
        self.alpha = alpha
        self.n_states = n_states
        self.n_actions = n_actions

    def update(self, s, a, r, s_next):
        best_next_value = np.max(self.Q[s_next-1]) #Max of Q(s',a') over a'
        self.Q[s-1, a-1] += self.alpha * (r + self.gamma * best_next_value - self.Q[s-1, a-1]) #Bellman update

    def get_policy(self):
        return np.argmax(self.Q, axis=1) + 1


def process_dataset(filename, n_states, n_actions, gamma, alpha, n_epochs):
    print(f"Processing {filename}")
    data = pd.read_csv(filename).values
    data[:, [0, 1, 3]] = data[:, [0, 1, 3]].astype(int)

    model = BatchQLearning(n_states, n_actions, gamma, alpha)

    for epoch in range(n_epochs):
        print(f'iter {epoch}/{n_epochs}')
        for s, a, r, s_next in data:
            model.update(s, a, r, s_next)

        # if epoch == n_epochs:
            # policy = model.get_policy()
            # print(f"Unique actions at epoch {epoch}: {np.unique(policy, return_counts=True)}") #sanity check that we're getting unique actions

    policy = model.get_policy()
    # save_policy_to_file(policy, filename)
    print('done with policy')
    return policy


def save_policy_to_file(policy, filename):
    policy_filename = filename.replace('.csv', '.policy')
    np.savetxt(policy_filename, policy, fmt='%d')
    print(f"Policy saved to {policy_filename}")
    print(f"Final unique actions in policy: {np.unique(policy, return_counts=True)}")


def main():
    files = ['data/small.csv',
             'medium.csv',
             'large.csv']

    #n_states, n_actions, gamma, alpha, n_epochs:
    params = [
        (100, 4, 0.95, 0.1, 100),
        (50000, 7, 1, 0.41, 100),
        (302020, 9, 0.95, 0.075, 100)
    ]

    for file, param in zip(files, params):
        process_dataset(file, *param)


if __name__ == '__main__':
    main()