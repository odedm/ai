# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import game

from game import Agent
from ghostAgents import GhostAgent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        oldCapsules = currentGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = 0
        
        # Try to reach the food
        score += 100 - min([util.manhattanDistance(fp, newPos) for fp in oldFood.asList()])
        
        # Try to reach the capsules
        if oldCapsules:
            score += 20 - min([util.manhattanDistance(cp, newPos) for cp in oldCapsules])
        
        # Find where the closest ghost is
        (closestGhost, idx) = min([ (util.manhattanDistance(gp, newPos), i) 
                                   for (i, gp) in enumerate( [agent.configuration.pos for agent in newGhostStates] ) ] )
        
        # If any ghosts are scared, try to eat them 
        if newScaredTimes[idx] > 2:
            score += 200 - closestGhost
        
        # If the ghosts are not scared and close by, try to avoid them
        else:        
            if closestGhost <= 0:
                score -= 150
            elif closestGhost <= 2:
                score -= 50
        
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    ans = currentGameState.getScore()
    #print(ans)
    return ans

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
    
        Here are some method calls that might be useful when implementing minimax.
    
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
    
        Directions.STOP:
        The stop direction, which is always legal
    
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
    
        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        action = self.MaxValue(gameState, 0, retAction=True)
        return action
    
    def MaxValue(self, state, depth, retAction=False):
        """
        Max player point of view - return the maximum value between
        the min players' values
        """
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        v = float('-inf')
        agent = 0
        chosenAction = None
        depth += 1
            
        for action in state.getLegalActions(agent):
            if action == Directions.STOP:
                # ignore STOP direction
                continue
            succ = state.generateSuccessor(agent, action)
            minval = self.MinValue(succ, depth, agent+1)
            if minval > v:
                v = minval
                chosenAction = action  
        
        if retAction:
            return chosenAction
                
        return v

    def MinValue(self, state, depth, agent):
        """
        Min player point of view -
        Last ghost agent returns the minimum value among pacman's moves.
        Other ghost agents return the minimum after passing control to the next agent.
        """
        if state.isWin() or state.isLose() or (agent == state.getNumAgents() - 1 and depth == self.depth):
            return self.evaluationFunction(state)
        
        v = float('inf')
            
        for action in state.getLegalActions(agent):
            succ = state.generateSuccessor(agent, action)
            if agent == state.getNumAgents() - 1:
                v = min(v, self.MaxValue(succ, depth))
            else:
                v = min(v, self.MinValue(succ, depth, agent+1))
        
        return v
        
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action = self.alphaBetaMaxValue(gameState, 0, float('-inf'), float('inf'), retAction=True)
        return action
    
    def alphaBetaMaxValue(self, state, depth, alpha, beta, retAction=False):
        """
        Max player point of view - return the maximum value between
        the min players' values, with pruning
        """
        
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        v = float('-inf')
        agent = 0
        chosenAction = None
        depth += 1
            
        for action in state.getLegalActions(agent):
            if action == Directions.STOP:
                continue
            succ = state.generateSuccessor(agent, action)
            minval = self.alphaBetaMinValue(succ, depth, agent+1, alpha, beta)
            if minval > v:
                v = minval
                chosenAction = action
            alpha = max(alpha, v)  
            if alpha >= beta:
                # prune
                return v
        
        if retAction:
            return chosenAction
                
        return v

    def alphaBetaMinValue(self, state, depth, agent, alpha, beta):
        """
        Min player point of view -
        Last ghost agent returns the minimum value among pacman's moves.
        Other ghost agents return the minimum after passing control to the next agent.
        Uses pruning.
        """
        if state.isWin() or state.isLose() or (agent == state.getNumAgents() - 1 and depth == self.depth):
            return self.evaluationFunction(state)
        
        v = float('inf')
            
        for action in state.getLegalActions(agent):
            succ = state.generateSuccessor(agent, action)
            
            if agent == state.getNumAgents() - 1:
                v = min(v, self.alphaBetaMaxValue(succ, depth, alpha, beta))
            else:
                v = min(v, self.alphaBetaMinValue(succ, depth, agent+1, alpha, beta))
                
            beta = min(beta, v)
            if alpha >= beta:
                # prune
                return v
        
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action = self.expectiMaxValue(gameState, 0, True)
        return action
    
    def expectiMaxValue(self, state, depth, retAction=False):
        """
        Max player point of view - return the maximum value between
        the min players' values.
        """
        
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        v = float('-inf')
        agent = 0
        chosenAction = None
        depth+=1
            
        for action in state.getLegalActions(agent):
            if action == Directions.STOP:
                continue
            succ = state.generateSuccessor(agent, action)
            minval = self.expectiMinValue(succ, depth, agent+1)
            if minval > v:
                v = minval
                chosenAction = action  
        
        if retAction:
            return chosenAction
                
        return v

    def expectiMinValue(self, state, depth, agent):
        """
        Min (random) player point of view -
        Last ghost agent returns the expected value among pacman's moves.
        Other ghost agents return the expected values after passing control to the next agent.
        """

        if state.isWin() or state.isLose() or (agent == state.getNumAgents() - 1 and depth == self.depth):
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(agent)
        
        tot = 0
        # Calculate expected value among moves
        for action in actions:
            succ = state.generateSuccessor(agent, action)
            if agent == state.getNumAgents() - 1:
                tot += (1./len(actions)) * self.expectiMaxValue(succ, depth)
            else:
                tot += (1./len(actions)) * self.expectiMinValue(succ, depth, agent+1)
        
        return tot
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    This function aimes to give a good evaluation of the current state,
    in one of two modes: 'scared' and 'not scared'.
    When we expect scared ghosts, we want to eat the capsule (which is nearby) and then
    eat the ghosts.
    When the only relevant objective is food, it gets the larger part of the score.
    Finishing all the food gets a crazy score, since we will alost always want to eat
    the last pellet.  
    the score is a linear combination of the different state features - 
    food distance, scared timers, capsule distance etc.
    """
    
    pos = currentGameState.getPacmanPosition()    
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    score = 0
    # current state score - good at helping ghost avoidance
    s = 5 * currentGameState.getScore()    
    
    if sum(scaredTimes) > 0:
        # Go to capsule (if close), eat scared ghosts
        scaredDists = [util.manhattanDistance(gp, pos) for gp in [g.configuration.pos for g in ghostStates if g.scaredTimer > 0]]
        gradeNumGhosts = 0
        if scaredDists:
            gradeNumGhosts = 100. / len(scaredDists)
        gradeScared = 1 * sum(scaredTimes)
        gradeCapsule = 10 * (1. / (len(capsules) + 1))
        score += gradeScared + gradeCapsule + gradeNumGhosts
    
    if food:
        # Go to food
        gradeFoodNum = 100 * (1. / (len(food)+1) )
        gradeFoodDist = 10 * 1. / (1 + ((sum([util.manhattanDistance(cp, pos) for cp in food]) / float(len(food)+1))))
        gradeNearFoodDist = 10 * 1. /(min([util.manhattanDistance(cp, pos) for cp in food]))
        
        score += s + gradeFoodNum + gradeFoodDist + gradeNearFoodDist
    
    else:
        # Always try to eat the last pellet
        score += 10000

    return score
        

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
