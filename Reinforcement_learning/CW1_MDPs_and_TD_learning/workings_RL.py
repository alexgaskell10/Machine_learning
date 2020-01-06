import numpy as np
import matplotlib.pyplot as plt

class GridWorld:

    def __init__(self, gamma=0.25, p=0.4, threshold=10**-6):
        self.gamma = gamma
        self.p = p
        self.states = 11
        self.threshold = threshold
        self.terminal_states = [1,10]

        self.dict_transitions, self.t_m = self.build_transition_matrix()
        self.reward_fun = self.get_rewards()

    def build_transition_matrix(self):
        '''
        :return dict_transitions: dictionary showing, for each grid, where the agent would end up if it moved in a direction
        :return transition_matrix: creates transition matrix (11x11x4 tensor) showing probability of moving to a different
        state from the agent's current state given an action. The action is stochastic with probability p of success.
        '''
        # Format: current_grid:[N,E,S,W]
        dict_transitions = {
            1:np.array([1,2,5,1]),
            2:np.array([2,2,2,2]),        # Terminal state
            3:np.array([3,4,3,2]),
            4:np.array([4,4,7,3]),
            5:np.array([1,6,5,5]),
            6:np.array([2,6,8,5]),
            7:np.array([4,7,10,7]),
            8:np.array([6,9,8,8]),
            9:np.array([9,10,11,8]),
            10:np.array([7,10,10,9]),
            11:np.array([11,11,11,11])       # Terminal state
        }

        transition_matrix = np.zeros((self.states,self.states,4))

        for k,vals in dict_transitions.items():
            for n,v in enumerate(vals):
                transition_matrix[k-1,v-1,n] += self.p        # p(transition|action)

                for i in range(1,4):
                    transition_matrix[k-1,v-1,(n+i)%4] += (1-self.p)/3        # p(transition|not action)

        assert (np.sum(transition_matrix, axis = 1) - 1 < 0.01).all(), "Probabilites do not sum to 1"

        return dict_transitions, transition_matrix

    def get_rewards(self):
        '''
        Assumes 0 reward for every state except the terminal states
        '''
        rewards = {1:-1,2:10,3:-1,4:-1,5:-1,6:-1,7:-1,8:-1,9:-1,10:-1,11:-100}
        return list(rewards.values())

    def compute_value_fn(self, policy, value_fn):
        '''
        Uses the Bellman equation to compute the value function for each state given a policy.
        Assumes that the transition cost is not discounted (& is and replaced by the rewards of the transition states
        if transitioning to the terminal_states)

        :return updated_value_fn: array containing new value function
        '''
        updated_value_fn = np.zeros((self.states,))

        for s in range(self.states):
            if s in self.terminal_states:
                updated_value_fn[s] = value_fn[s]
            else:
                transition_ps = np.sum([p*self.t_m[s,:,n] for n,p in enumerate(policy[s])], axis=0)
                val = sum([t_p*(R + self.gamma*val) if R not in [-100, 10] else t_p*R for t_p,val,R in zip(transition_ps, value_fn, self.reward_fun)])      # Bellman equation
                updated_value_fn[s] = val

        return updated_value_fn

    def policy_evaluation(self, policy, value_fn):
        '''
        Iteratively evaluates a given policy until the value function stabilises

        :return value_fn: updated value function evaluated at the policy
        '''
        delta = 2*self.threshold        # Init delta at above the threshold

        while delta > self.threshold:

            value_fn_old = value_fn.copy()
            value_fn = self.compute_value_fn(policy, value_fn)

            delta = max(abs(value_fn-value_fn_old))

        return value_fn

    def improve_policy(self, policy, value_fn):
        '''
        Given a value function, compute the optimal policy (i.e. for each state, find the
        adjacent state with the highest value, and set the policy to moving to this cell).

        :return P: an optimal policy (11x4 array) given the value function
        '''
        P = policy.copy()
        for i,p in enumerate(list(self.dict_transitions.values())):
            if i not in self.terminal_states:
                argmax_p = np.argmax([value_fn[int(j)-1] for j in p])
                P[i] = np.zeros((4,))
                P[i,argmax_p] = 1
        return P

    def policy_iteration(self):
        '''
        Perform policy iteration to find an optimal policy and optimal state value function.

        Method:
        - Init the policy randomly and the value function to the state rewards
        - Perform policy evaluation to get value function given policy
        - Perform policy improvement to get optimal policy given value function
        - Repeat the above 2 steps until the state function stabilises

        :return policy: optimal policy (11x4)
        :return value_fn: optimal state value function (11,)
        :return epochs: # iterations until convergence
        '''
        policy = np.ones((11,4))/4      # Begin with a random policy
        value_fn = np.array(self.reward_fun)        # Init value function to the rewards

        delta = 2*self.threshold        # Init delta at above the threshold

        epochs = 0
        while delta > self.threshold:
            value_fn_old = value_fn.copy()      # Save old val_fn for checking convergence
            value_fn = self.policy_evaluation(policy, value_fn)
            policy = self.improve_policy(policy, value_fn)

            epochs += 1     # Track convergence epochs

            delta = max(abs(value_fn-value_fn_old))

        for i in self.terminal_states:         # Hard code the values of the terminal states to 0 (very hacky, sorry! But needed to make my code work...)
            value_fn[i] = 0

        return policy, value_fn, epochs

    def main(self):
        '''
        For running GridWorld methods
        '''
        optimal_policy, optimal_value_fn, epochs = self.policy_iteration()
        print(optimal_policy)
        print(optimal_value_fn)
        print(epochs)


if __name__ == '__main__':
    np.set_printoptions(precision = 3)

    grid = GridWorld()
    # grid = GridWorld(p = 0.9999999, gamma=0.6)
    # grid = GridWorld(gamma=0.5, p=1, threshold=10**-6)
    grid.main()
