import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon = 0.1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # This just formulates the epsilon-greedy policy here
        
        policy_s = np.ones(self.nA) * self.epsilon / self.nA # this is exploration
        best_a = np.argmax(self.Q[state]) # action with the max expected q value
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA) # exploitation
        return np.random.choice(np.arange(self.nA), p = policy_s) , policy_s

    def step(self, state, action, reward, next_state, next_action,policy_in, done, flag):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment 
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha = 0.1
        gamma= 0.95
        # This implements the q learning algorithm
        if not done:
            
            if (flag==1):   # q learning
                self.Q[state][action] += alpha * (reward + (gamma * np.max(self.Q[next_state])) - self.Q[state][action])
            
            elif (flag==2): # sarsa here
                self.Q[state][action] += alpha * (reward + (gamma *self.Q[next_state][next_action]) - self.Q[state][action])
        
            elif(flag==3):  # Expected sarsa here
                self.Q[state][action] += alpha * (reward + (gamma*sum(policy_in*self.Q[next_state])) - self.Q[state][action])

            #print('Not done, updating Q table here')
        if done:
            self.epsilon = max(self.epsilon*0.99, 0.00005) # this updates the epsilon here
            #print(self.epsilon)
            