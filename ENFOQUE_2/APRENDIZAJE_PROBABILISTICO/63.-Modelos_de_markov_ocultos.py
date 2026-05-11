import numpy as np

class HiddenMarkovModel:
    def __init__(self, states, observations, trans_prob, emit_prob, initial_prob):
        self.states = states
        self.observations = observations
        self.trans_prob = trans_prob  # Transition probabilities: dict of dicts
        self.emit_prob = emit_prob    # Emission probabilities: dict of dicts
        self.initial_prob = initial_prob  # Initial probabilities: dict

    def viterbi(self, obs_seq):
        """
        Viterbi algorithm to find the most likely sequence of hidden states.
        """
        n_states = len(self.states)
        n_obs = len(obs_seq)
        viterbi = np.zeros((n_states, n_obs))
        backpointer = np.zeros((n_states, n_obs), dtype=int)

        # Initialization
        for i, state in enumerate(self.states):
            viterbi[i, 0] = self.initial_prob[state] * self.emit_prob[state][obs_seq[0]]
            backpointer[i, 0] = 0

        # Recursion
        for t in range(1, n_obs):
            for j, state_j in enumerate(self.states):
                max_prob = -np.inf
                max_state = 0
                for i, state_i in enumerate(self.states):
                    prob = viterbi[i, t-1] * self.trans_prob[state_i][state_j] * self.emit_prob[state_j][obs_seq[t]]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                viterbi[j, t] = max_prob
                backpointer[j, t] = max_state

        # Termination
        best_path_prob = np.max(viterbi[:, -1])
        best_last_state = np.argmax(viterbi[:, -1])

        # Backtrack
        best_path = [best_last_state]
        for t in range(n_obs-1, 0, -1):
            best_last_state = backpointer[best_last_state, t]
            best_path.insert(0, best_last_state)

        return [self.states[i] for i in best_path], best_path_prob

# Example usage
if __name__ == "__main__":
    states = ['Sunny', 'Rainy']
    observations = ['Walk', 'Shop', 'Clean']
    trans_prob = {
        'Sunny': {'Sunny': 0.8, 'Rainy': 0.2},
        'Rainy': {'Sunny': 0.4, 'Rainy': 0.6}
    }
    emit_prob = {
        'Sunny': {'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1},
        'Rainy': {'Walk': 0.1, 'Shop': 0.4, 'Clean': 0.5}
    }
    initial_prob = {'Sunny': 0.6, 'Rainy': 0.4}

    hmm = HiddenMarkovModel(states, observations, trans_prob, emit_prob, initial_prob)
    obs_seq = ['Walk', 'Shop', 'Clean']
    path, prob = hmm.viterbi(obs_seq)
    print("Most likely state sequence:", path)
    print("Probability:", prob)