import numpy as np
import matplotlib.pyplot as plt

'''2. Write a funciton that returns the state transition probability'''
# Note the below means the probability of transitioning FROM state s
# TO state'
def state_transitions(i):
    s_ts = np.array([
        np.array([0.9, 0.1, 0, 0, 0, 0, 0]), #Facebook
        np.array([0.5, 0, 0.5, 0, 0, 0, 0]), # Class 1
        np.array([0, 0, 0, 0.8, 0, 0, 0.2]), # Class 2
        np.array([0, 0, 0, 0, 0.4, 0.6, 0]), # Class 3
        np.array([0, 0.2, 0.4, 0.4, 0, 0, 0]), # Pub
        np.array([0, 0, 0, 0, 0, 0, 1]), # Pass
        np.array([0, 0, 0, 0, 0, 0, 1]) # Sleep
    ])
    return s_ts[i]

'''3. Write a function that gives you Rs (note that in our
MRP the reward is deterministic, so the expectation is the
immediate reward)'''
def reward(i):
    # Returns immediate expected reward in state i
    r = gamma * state_transitions(i).reshape(1,n) @ S
    return r

'''4. Sample a trace (the sequence of state, reward, state, reward
,..., terminal state, 0) from this MRP.
Hint: you need to write
functions that e.g. imnplement the probabilistic state transition
dynamics. Simulate a run of the MRP always using Class1 as the only
initial state.'''
def trace(s_t):
    journey = [(s_t, int(S[s_t]))]
    while s_t != 6:
        # Get next state
        s_t = int(np.random.choice(n, 1, p = state_transitions(s_t)))
        # Compute reward at next state
        r_t = int(S[s_t])
        # Save reward and state
        journey.append((s_t, r_t))
    # print(journey)
    return journey

'''5. Write a function that computes the return of a specific trace
(i.e. from its initial state till it reaches the terminal state).'''
def expec_reward(journey):
    rs = [j[1] * gamma**n for n,j in enumerate(journey)]
    return sum(rs)

'''6. Write a function that computes (by averaging over the returns
of many sampled traces) the value of each state in our MRP.
This is a way of computing the state value function (Why?)'''
def compute_state_value_fn(samples):
    avg_rewards = []
    rewards = []
    for s_t in range(n): #[3]:
        reward = [expec_reward(trace(s_t)) for i in range(samples)]
        avg_rewards.append(sum(reward)/samples)
        rewards.append(reward)
    return avg_rewards, rewards

'''How many repeated samples do you need to get a good estimate of
the state value. Plot for the state Class1 the empirical average
of the returns of individual traces as a function of the number of runs.'''
def plot_sample_params(samples):
    avg_rewards, rewards = compute_state_value_fn(samples)
    avg_culm_reward = [[sum(k[:i])/i for i in range(1, samples+1)]
                        for k in rewards]
    for i in avg_culm_reward:
        plt.plot(i)[10:]
    plt.legend([s[0] for s in States])
    plt.show()

def main():
    plot_sample_params(samples)

if __name__ == '__main__':
    '''1. Choose simple code representation of the S state
    space and implement it'''
    States = [('Facebook',-1), ('Class 1',-2), ('Class 2',-2), ('Class 3',-2),
              ('Pub',1), ('Pass',10), ('Sleep',0)]
    # Return States in format for matrix multiplication
    n = len(States)
    S = np.array([i[1] for i in States]).reshape(n, 1)

    gamma = 0.9
    samples = 500

    main()
