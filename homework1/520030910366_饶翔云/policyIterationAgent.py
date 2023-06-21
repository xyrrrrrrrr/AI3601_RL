import util
from abstractAgent import Agent
import random

class PolicyIterationAgent(Agent):
    """An agent that takes a Markov decision process on initialization
    and runs policy iteration for a given number of iterations.

    Hint: Test your code with commands like `python main.py -a policy -i 100 -k 10`.
    """

    def __init__(self, mdp, discount = 0.9, epsilon=0.001, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of policy iteration
        self.iterations = iterations # The policy iteration will run AT MOST these steps
        self.values = util.Counter() # You need to keep the record of all state values here
        self.policy = dict()
        self.runPolicyIteration()

    def runPolicyIteration(self):
        """ YOUR CODE HERE """
        # calculate V according to the policy
        # initialize the policy
        for state in self.mdp.getStates():
            # If the state is terminal, then the policy is None
            if self.mdp.isTerminal(state):
                self.policy[state] = None
            else:
            # Otherwise, the policy is the one of the possible actions
                actions = self.mdp.getPossibleActions(state)
                idx = random.randint(0, len(actions)-1)
                # print(actions,idx)
                self.policy[state] = actions[idx]
        current_policy = self.policy
        for _ in range(0, self.iterations):
            while True:
                new_values = util.Counter()
                for state in self.mdp.getStates():
                    # If the state is terminal, then the value is 0
                    if self.mdp.isTerminal(state):
                        new_values[state] = 0
                    else:
                    # Otherwise, the value is the maximum Q of all possible actions(using old values)
                        action = current_policy[state]
                        new_values[state] = self.computeQValueFromValues(state, action)
                # If the maximum change is less than epsilon, then stop
                change = [abs(self.values[state] - new_values[state]) for state in self.mdp.getStates()]
                # Update the values
                self.values = new_values
                if max(change) < self.epsilon:
                    break
            # calculate the new policy
            new_policy = dict()
            for state in self.mdp.getStates():
                # If the state is terminal, then the policy is None
                if self.mdp.isTerminal(state):
                    new_policy[state] = None
                else:
                # Otherwise, the policy is the action with the maximum Q
                    actions = self.mdp.getPossibleActions(state)
                    new_policy[state] = max(actions, key=lambda action: self.computeQValueFromValues(state, action))
            # If the new policy is the same as the old policy, then stop
            if new_policy == current_policy:
                break
            else:
                current_policy = new_policy
        # Update the policy
        self.policy = current_policy


    def getValue(self, state):
        """Return the value of the state (computed in __init__)."""
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """Compute the Q-value of action in state from the value function stored in self.values."""

        value = None

        """ YOUR CODE HERE """
        # The Q-value is the sum of all possible next states and their probabilities
        value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])

        return value

    def computeActionFromValues(self, state):
        """The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        bestaction = None

        """ YOUR CODE HERE """
        # If the state is terminal, then there is no action
        if not self.mdp.isTerminal(state):
            # Find the action with the maximum Q
            actions = self.mdp.getPossibleActions(state)
            bestaction = max(actions, key=lambda action: self.computeQValueFromValues(state, action))

        return bestaction

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)