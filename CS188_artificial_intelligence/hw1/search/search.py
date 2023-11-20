# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    "*** YOUR CODE HERE ***"

    #initialize the actions recorded
    actions = []
    #record the parents of every point
    parents = {}
    #record the point visited
    visited = []
    #initialize the start state
    cur_state = (problem.getStartState(), "start", 0)
    #build a stack to store state
    states = util.Stack()
    #first push the start state
    states.push(cur_state)

    while(states.isEmpty() == False):
        #pop the cur_state
        cur_state = states.pop()
        visited.append(cur_state[0])
        if(problem.isGoalState(cur_state[0])):
            #print(actions)
            break
        #get all possible successors
        successors = problem.getSuccessors(cur_state[0])

        #visit all successors
        for next_step in successors:
            if(next_step[0] not in visited):
                #push next_step into stack
                states.push(next_step)
                parents[next_step[0]] = cur_state

    while(cur_state[0] != problem.getStartState()):
        actions.append(cur_state[1])
        parent = parents[cur_state[0]]
        cur_state = parent

    return actions[::-1]

    '''
    path = []
    action = actions[-1]
    print(action)
    while(action[0] != problem.getStartState()):
        path.append(action[1])
        ori_loc = action[0]
        #print(ori_loc)
        if(action[1] == "North"):
            new_loc = (ori_loc[0]+0,ori_loc[1]-1)
        elif(action[1] == "South"):
            new_loc = (ori_loc[0]+0,ori_loc[1]+1)
        elif(action[1] == "West"):
            new_loc = (ori_loc[0]+1,ori_loc[1]+0)
        else:
            new_loc = (ori_loc[0]-1,ori_loc[1]+0)

        if(new_loc == problem.getStartState()):
            break

        for pos in actions:
            if(pos[0] == new_loc):
                action = pos
                print(action[0],action[1])

    reversed_path = []
    for i in range(len(path)):
        reversed_path.append(path[len(path)-i-1])
    #print(reversed_path)
    return reversed_path
    '''


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #initialize the actions recorded
    actions = []
    #record the parents of every point
    parents = {}
    #record the point visited
    visited = []
    #initialize the start state
    cur_state = (problem.getStartState(), "start", 0)
    #build a queue to store state
    states = util.Queue()
    #first push the start state
    states.push(cur_state)
    visited.append(cur_state[0])

    while(states.isEmpty() == False):
        #pop the cur_state
        cur_state = states.pop()
        if(problem.isGoalState(cur_state[0])):
            #print(actions)
            break
        #get all possible successors
        successors = problem.getSuccessors(cur_state[0])

        #visit all successors
        for next_step in successors:
            #print(next_step)
            if(next_step[0] not in visited):
                #push next_step into stack
                states.push(next_step)
                '''
                different from DFS
                in BFS we must record the point which has been enqueued
                so that we will not enqueue same point twice
                '''
                visited.append(next_step[0])
                parents[next_step[0]] = cur_state

    while(cur_state[0] != problem.getStartState()):
        actions.append(cur_state[1])
        parent = parents[cur_state[0]]
        cur_state = parent

    return actions[::-1]

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #initialize the actions recorded
    actions = []
    #record the parents of every point
    parents = {}
    #record the point visited, but different from BFS, now we
    #need to record the point pop
    #when point pop, meaning no closer way, no need to push again
    visited = []
    #initialize the start state
    cur_state = problem.getStartState()
    #build a queue to store state
    states = util.PriorityQueue()
    #first push the start state
    states.push(cur_state, 0)
    #record the least cost
    least_cost = {}
    least_cost[cur_state] = 0

    while(states.isEmpty() == False):

        #pop the cur_state
        cur_state = states.pop()
        #print(cur_state, priority)
        #get the cost to cur_state
        origin_cost = least_cost[cur_state]

        if problem.isGoalState(cur_state):
            break

        visited.append(cur_state)

        #different from BFS
        #now we stop searching only when Priority is empty
        #get all possible successors
        successors = problem.getSuccessors(cur_state)

        #visit all successors

        for next_step in successors:
            if(next_step[0] not in visited):
                new_cost = origin_cost + next_step[2]
                states.update(next_step[0], new_cost)
                if(next_step[0] not in least_cost.keys() or new_cost < least_cost[next_step[0]]):
                    parents[next_step[0]] = (cur_state, next_step[1])
                    least_cost[next_step[0]] = new_cost

    while(cur_state != problem.getStartState()):
        actions.append(parents[cur_state][1])
        parent = parents[cur_state][0]
        cur_state = parent

    return actions[::-1]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #initialize the actions recorded
    actions = []
    #record the parents of every point
    parents = {}
    #record the point visited, but different from BFS, now we
    #need to record the point pop
    #when point pop, meaning no closer way, no need to push again
    visited = []
    #initialize the start state
    cur_state = problem.getStartState()
    #build a queue to store state
    states = util.PriorityQueue()
    #first push the start state
    states.push(cur_state, heuristic(cur_state, problem))
    #record the least cost
    least_cost = {}
    least_cost[cur_state] = 0

    while(states.isEmpty() == False):

        #pop the cur_state
        cur_state = states.pop()
        #get the cost to cur_state
        origin_cost = least_cost[cur_state]

        if problem.isGoalState(cur_state):
            break

        visited.append(cur_state)

        #different from BFS
        #now we stop searching only when Priority is empty
        #get all possible successors
        successors = problem.getSuccessors(cur_state)

        #visit all successors

        for next_step in successors:
            if(next_step[0] not in visited):
                new_cost = origin_cost + next_step[2] + \
                    heuristic(next_step[0], problem) - \
                    heuristic(cur_state, problem)
                states.update(next_step[0], new_cost)
                if(next_step[0] not in least_cost.keys() or new_cost < least_cost[next_step[0]]):
                    parents[next_step[0]] = (cur_state, next_step[1])
                    least_cost[next_step[0]] = new_cost

    while(cur_state != problem.getStartState()):
        actions.append(parents[cur_state][1])
        parent = parents[cur_state][0]
        cur_state = parent

    return actions[::-1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
