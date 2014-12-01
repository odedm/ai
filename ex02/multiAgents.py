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
        
        score += 100 - min([util.manhattanDistance(fp, newPos) for fp in oldFood.asList()])
        if oldCapsules:
            score += 20 - min([util.manhattanDistance(cp, newPos) for cp in oldCapsules])
                
        (closestGhost, idx) = min([ (util.manhattanDistance(gp, newPos), i) 
                                   for (i, gp) in enumerate( [agent.configuration.pos for agent in newGhostStates] ) ] )

        if newScaredTimes[idx] > 2:
            score += 200 - closestGhost
            
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
        v, action = self.MaxValue(gameState, 0, retAction=True)
        return action
    
    def MaxValue(self, state, depth, retAction=False):
        
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
            minval = self.MinValue(succ, depth, agent+1)
            if minval > v:
                v = minval
                chosenAction = action  
        
        if retAction:
            return v, chosenAction
                
        return v

    def MinValue(self, state, depth, agent):

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
        v, action = self.alphaBetaMaxValue(gameState, 0, float('-inf'), float('inf'), retAction=True)
        return action
    
    def alphaBetaMaxValue(self, state, depth, alpha, beta, retAction=False):
        
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
                return v
        
        if retAction:
            return v, chosenAction
                
        return v

    def alphaBetaMinValue(self, state, depth, agent, alpha, beta):

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
        v, action = self.expectiMaxValue(gameState, 0, True)
        return action
    
    def expectiMaxValue(self, state, depth, retAction=False):
        
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
            return v, chosenAction
                
        return v

    def expectiMinValue(self, state, depth, agent):

        if state.isWin() or state.isLose() or (agent == state.getNumAgents() - 1 and depth == self.depth):
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(agent)
        
        tot = 0
        for action in actions:
            succ = state.generateSuccessor(agent, action)
            if agent == state.getNumAgents() - 1:
                tot += (1./len(actions)) * self.expectiMaxValue(succ, depth)
            else:
                tot += (1./len(actions)) * self.expectiMinValue(succ, depth, agent+1)
        
        return tot
        
class Kruskal(object):
    """ This class implements the kruskal algorithm """
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, vertice):
        self.parent[vertice] = vertice
        self.rank[vertice] = 0
    
    def find(self, vertice):
        if self.parent[vertice] != vertice:
            self.parent[vertice] = self.find(self.parent[vertice])
        return self.parent[vertice]
    
    def union(self, vertice1, vertice2):
        root1 = self.find(vertice1)
        root2 = self.find(vertice2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]: self.rank[root2] += 1
    
    def kruskal(self, graph):
        """ Returns the MST of the given graph """
        for vertice in graph.vertices:
            self.make_set(vertice)
    
        mst = set()
        edges = list(graph.edges)
        edges.sort()
        for edge in edges:
            weight, vertice1, vertice2 = edge
            if self.find(vertice1) != self.find(vertice2):
                self.union(vertice1, vertice2)
                mst.add(edge)
        return mst


class Graph(object):
    """ Represents an abstract graph.
    Example graph:
    graph = {
        vertices: ['A', 'B', 'C', 'D', 'E', 'F'],
        edges: set([(1, 'A', 'B'), (5, 'A', 'C') ])
        }
    """
    
    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = set()
        
    def add_edge(self, v1, v2, w):
        if v1 != v2:
            self.edges.add((w, v1, v2))


def get_mst(foodList):
    """
    Builds an MST from the food list, using food coordinates
    as nodes in a fully-connected graph, with Manhattan distances acting as weights
    between every two nodes.
    """
    
    k = Kruskal()
    g = Graph(foodList)
    for pos in foodList:
        for pos2 in foodList:
            g.add_edge(pos, pos2, util.manhattanDistance(pos, pos2))
    
    return k.kruskal(g)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    pos = currentGameState.getPacmanPosition()    
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    score = 0
    s = 5 * currentGameState.getScore()    
    
    if sum(scaredTimes) > 0:
        scaredDists = [util.manhattanDistance(gp, pos) for gp in [g.configuration.pos for g in ghostStates if g.scaredTimer > 0]]
#         if scaredDists:
#             distanceToClosestScared = min(scaredDists)        
#         gradeDistScared = 100 -(distanceToClosestScared)
        gradeNumGhosts = 0
        if scaredDists:
            gradeNumGhosts = 100. / len(scaredDists)
        gradeScared = 1 * sum(scaredTimes)
        gradeCapsule = 10 * (1. / (len(capsules) + 1))
        score += gradeScared + gradeCapsule + gradeNumGhosts
    
    if food:
        gradeFoodNum = 100 * (1. / (len(food)+1) )
        gradeFoodDist = 10 * 1. / (1 + ((sum([util.manhattanDistance(cp, pos) for cp in food]) / float(len(food)+1))))
        gradeFarFoodDist = 0 * (max([util.manhattanDistance(cp, pos) for cp in food]))
        gradeNearFoodDist = 10 * 1. /(min([util.manhattanDistance(cp, pos) for cp in food]))
        
        score += s + gradeFoodNum + gradeFoodDist + gradeFarFoodDist + gradeNearFoodDist
    
    else:
        score += 10000

#     r = random.random()
#     if r <= 0.1:
#         score += 1000 
      
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
