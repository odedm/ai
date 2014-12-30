# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        #print(self.mdp.getTransitionStatesAndProbs(self.mdp.getStates()[3], 'north'))
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        states = self.mdp.getStates()

        for state in self.mdp.getStates():
            # Check that None is valid here!!
            # For when reward != 0
            self.values[state] = self.mdp.getReward(state, None, None)

        for i in range(1, iterations):
            tmpValues = self.values.copy()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    continue
                possibleActions = self.mdp.getPossibleActions(s)
                for a in possibleActions:
                    for s2, p in self.mdp.getTransitionStatesAndProbs(s,a):
                        self.values[s] += self.discount * p * tmpValues[s2]
                if len(possibleActions) > 0:
                    self.values[s] /= len(possibleActions)
                self.values[s] += self.mdp.getReward(s, None, None)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """

        import pdb; pdb.set_trace()
        tot = 0
        for s2, p in self.mdp.getTransitionStatesAndProbs(state, action):
            tot += p * self.values[s2]
        return tot

    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        ac = util.Counter()
        for a in self.mdp.getPossibleActions(state):
            for s2, p in self.mdp.getTransitionStatesAndProbs(state, a):
                ac[a] += p * self.values[s2]
        return ac.argMax()

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
