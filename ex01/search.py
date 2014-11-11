# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueueWithFunction
import pdb

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]


class fringeState(object):
    """ Every item in the fringe is a pair of position & path """
    def __init__(self, state, actions):
        self.state = state
        self.actions = actions
    



    
def getActions(problem, fringe):
    """ Returns the list of actions to reach the goal,
    Based on a given fringe, Fringe has to implement push, pop, isEmpty """
    # Start with start state in fringe, and no visited nodes.    
    fringe.push( fringeState(problem.getStartState(), []) )
    visited = set()
    
    while not fringe.isEmpty():
        current = fringe.pop()

        if current.state in visited:
            continue
        
        visited.add(current.state)
        
        if problem.isGoalState(current.state):
            # Reached goal
            return current.actions

        # Add all current's children to fringe
        for succ in problem.getSuccessors(current.state):
            fringe.push( fringeState(succ[0], current.actions + [succ[1]]) )
                        
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """    
    fringe = Stack()
    return getActions(problem, fringe)

def breadthFirstSearch(problem):
    """ Search the shallowest nodes in the search tree first. [p 81] """
    fringe = Queue()
    return getActions(problem, fringe)

def uniformCostSearch(problem):
    """ Search the node of least total cost first. """
    fringe = PriorityQueueWithFunction(getCostFunc(problem))
    return getActions(problem, fringe)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def getCostFunc(problem, heuristic=nullHeuristic):
    """ Returns a cost function for the given problem """
    def getCost(fstate):
        return problem.getCostOfActions(fstate.actions) + heuristic(fstate.state, problem)
    return getCost
    
def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    fringe = PriorityQueueWithFunction(getCostFunc(problem, heuristic))
    return getActions(problem, fringe)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
