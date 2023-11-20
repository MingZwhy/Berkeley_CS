# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print("successorGameState: ", successorGameState)
        newPos = successorGameState.getPacmanPosition()
        #print("newPos: ", newPos)
        newFood = successorGameState.getFood()
        #print("newFood: \n", newFood)
        newGhostStates = successorGameState.getGhostStates()
        #print("newGhostStates: \n", newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print("newScaredTimes: ", newScaredTimes)
        "*** YOUR CODE HERE ***"

        #to evaluate a new state
        #we should consider two aspects of new state:
        #1: food
        #2: ghost

        #for food: we must consider both the number of food left and the shortest
        #distance between pacman and food
        #for ghost: we should consider the shortest
        #distance between pacman and ghost

        #so Score = score(food) + score(ghost)
        #         = score(food number, shortest dis) + score(shortest dis)

        #store locations of foods in term of list
        food_locs = newFood.asList()
        #get number of food
        food_number = len(food_locs)
        #get the location of food nearest pacman

        #use manhattan_distance to evluate the distance

        def manhattan_distance(loc):
            x1, y1 = newPos   #loc of pacman
            x2, y2 = loc      #loc of food
            return (abs(x2-x1) + abs(y2-y1))

        shortest_food_dis = 0

        if(food_number == 0):
            shortest_food_dis = 0
        else:
            food_locs = sorted(food_locs, key=lambda loc: manhattan_distance(loc))
            shortest_food_dis = manhattan_distance(food_locs[0])

        #get dangerous ghosts
        danger_ghosts_loc = []
        for ghost in newGhostStates:
            #if ghost.scaredtime != 0
            #we can ignore that ghost
            if(ghost.scaredTimer == 0):
                loc = ghost.getPosition()
                danger_ghosts_loc.append(loc)

        shortest_ghost_dis = 0

        if(len(danger_ghosts_loc) == 0):
            shortest_ghost_dis = 0
        else:
            danger_ghosts_loc = sorted(danger_ghosts_loc, key=lambda loc: manhattan_distance(loc))
            shortest_ghost_dis = manhattan_distance(danger_ghosts_loc[0])

        # when shortest_ghost_dis <= 2, it is very dangerous
        # we mush avoid this situation!!!! so return -10000
        if(shortest_ghost_dis <= 2):
            return -10000

        # less food, shorter shortest_food_dis and further ghost_dis <==> better loc
        sigma1 = -1
        sigma2 = -10
        sigma3 = 1

        score = sigma1 * shortest_food_dis + sigma2 * food_number + sigma3 * shortest_ghost_dis
        return score

        #return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        best_action = self.agent_action(gameState, 1)
        return best_action

    def judge_end(self, gameState):
        return (gameState.isWin() or gameState.isLose())

    def agent_action(self, gameState, depth):
        # agent —— for max

        # first we need to judge if win or lose
        if(self.judge_end(gameState)):
            return self.evaluationFunction(gameState)

        # initialize the max value
        max_val = -10000
        # initialize the best action
        best_action = Directions.STOP

        # get legal actions of agent (index=0)
        agent_actions = gameState.getLegalActions(0)

        # try every action in legal agent actions
        for action in agent_actions:
            # get next game state of this action
            next_gamestate = gameState.generateSuccessor(0, action)
            first_ghost_index = 1
            final_val = self.ghost_action(next_gamestate, depth, first_ghost_index)

            # update max_val and best action
            if final_val > max_val:
                max_val = final_val
                best_action = action

        # if top layer --> return best action
        # else return value
        if(depth == 1):
            return best_action
        else:
            return max_val

    def ghost_action(self, gameState, depth, index_of_ghost):

        # first we need to judge if win or lose
        if(self.judge_end(gameState)):
            return self.evaluationFunction(gameState)

        # initialize the min value
        min_val = 10000
        # number of ghosts
        ghosts_number = gameState.getNumAgents() - 1

        # get legal actions of ghost (index = index_of_ghost)
        ghost_actions = gameState.getLegalActions(index_of_ghost)

        # try every action in legal ghost actions
        for action in ghost_actions:
            # divide into two situation
            # 1: this is not last ghost, then need to min next ghost_action
            # 2: this is last ghost, then need to max agent action

            next_gamestate = gameState.generateSuccessor(index_of_ghost, action)

            if(index_of_ghost < ghosts_number):
                # explore next ghost, same depth
                next_ghost_index = index_of_ghost + 1
                final_val = self.ghost_action(next_gamestate, depth, next_ghost_index)

                # update min_val
                if(final_val < min_val):
                    min_val = final_val

            else:
                # explore next layer --> agent
                # also divide into two situations
                # 2.1: reach bottom layer --> return score
                # 2.2: still in the mid layer --> return value from agent

                if(depth == self.depth):
                    bottom_val = self.evaluationFunction(next_gamestate)

                    # update min_val
                    if(bottom_val < min_val):
                        min_val = bottom_val

                else:
                    final_val = self.agent_action(next_gamestate, depth+1)

                    # update min_val
                    if(final_val < min_val):
                        min_val = final_val

        return min_val




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -100000
        beta = 100000

        best_action = self.agent_action(gameState, 1, alpha, beta)
        return best_action

    def judge_end(self, gameState):
        return (gameState.isWin() or gameState.isLose())

    def agent_action(self, gameState, depth, alpha, beta):
        # agent —— for max

        # first we need to judge if win or lose
        if(self.judge_end(gameState)):
            return self.evaluationFunction(gameState)

        # initialize the max value
        max_val = -10000
        # initialize the best action
        best_action = Directions.STOP

        # get legal actions of agent (index=0)
        agent_actions = gameState.getLegalActions(0)

        # try every action in legal agent actions
        for action in agent_actions:
            # get next game state of this action
            next_gamestate = gameState.generateSuccessor(0, action)
            first_ghost_index = 1
            final_val = self.ghost_action(next_gamestate, depth, first_ghost_index, alpha, beta)

            # update max_val and best action
            if final_val > max_val:
                max_val = final_val
                best_action = action

            # judge whether or not to prune
            # if max_val > beta : no need to explore more nodes
            if max_val > beta:
                # no need to explore more nodes
                return max_val

            # update alpha
            if max_val > alpha:
                alpha = max_val


        # if top layer --> return best action
        # else return value
        if(depth == 1):
            return best_action
        else:
            return max_val


    def ghost_action(self, gameState, depth, index_of_ghost, alpha, beta):

        # first we need to judge if win or lose
        if(self.judge_end(gameState)):
            return self.evaluationFunction(gameState)

        # initialize the min value
        min_val = 10000
        # number of ghosts
        ghosts_number = gameState.getNumAgents() - 1

        # get legal actions of ghost (index = index_of_ghost)
        ghost_actions = gameState.getLegalActions(index_of_ghost)

        # try every action in legal ghost actions
        for action in ghost_actions:
            # divide into two situation
            # 1: this is not last ghost, then need to min next ghost_action
            # 2: this is last ghost, then need to max agent action

            next_gamestate = gameState.generateSuccessor(index_of_ghost, action)

            if(index_of_ghost < ghosts_number):
                # explore next ghost, same depth
                next_ghost_index = index_of_ghost + 1
                final_val = self.ghost_action(next_gamestate, depth, next_ghost_index, alpha, beta)

                # update min_val
                if(final_val < min_val):
                    min_val = final_val

            else:
                # explore next layer --> agent
                # also divide into two situations
                # 2.1: reach bottom layer --> return score
                # 2.2: still in the mid layer --> return value from agent

                if(depth == self.depth):
                    bottom_val = self.evaluationFunction(next_gamestate)

                    # update min_val
                    if(bottom_val < min_val):
                        min_val = bottom_val

                else:
                    final_val = self.agent_action(next_gamestate, depth+1, alpha, beta)

                    # update min_val
                    if(final_val < min_val):
                        min_val = final_val

            # judge whether or not to prune
            # if min_val < alpha : no need to explore more nodes
            if min_val < alpha:
                # no need to explore more nodes
                return min_val

            # update the beta
            if min_val < beta:
                beta = min_val

        return min_val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action = self.agent_action(gameState, 1)
        return action

    def judge_end(self, gameState):
        return (gameState.isWin() or gameState.isLose())

    def agent_action(self, gameState, depth):
        # agent —— for max

        # first we need to judge if win or lose
        if(self.judge_end(gameState)):
            return self.evaluationFunction(gameState)

        # initialize the max value
        max_val = -10000
        # initialize the best action
        best_action = Directions.STOP

        # get legal actions of agent (index=0)
        agent_actions = gameState.getLegalActions(0)

        # try every action in legal agent actions
        for action in agent_actions:
            # get next game state of this action
            next_gamestate = gameState.generateSuccessor(0, action)
            first_ghost_index = 1
            final_val = self.random_ghost_action(next_gamestate, depth, first_ghost_index)

            # update max_val and best action
            if final_val > max_val:
                max_val = final_val
                best_action = action

        # if top layer --> return best action
        # else return value
        if(depth == 1):
            return best_action
        else:
            return max_val


    def random_ghost_action(self, gameState, depth, index_of_ghost):

        # first we need to judge if win or lose
        if(self.judge_end(gameState)):
            return self.evaluationFunction(gameState)

        # initialize the min value
        min_val = 10000
        # number of ghosts
        ghosts_number = gameState.getNumAgents() - 1

        # get legal actions of ghost (index = index_of_ghost)
        ghost_actions = gameState.getLegalActions(index_of_ghost)

        # average value
        average_value = 0.0
        # weight
        weight = 1.0 / len(ghost_actions)

        # try every action in legal ghost actions
        for action in ghost_actions:
            # divide into two situation
            # 1: this is not last ghost, then need to min next ghost_action
            # 2: this is last ghost, then need to max agent action

            next_gamestate = gameState.generateSuccessor(index_of_ghost, action)

            if(index_of_ghost < ghosts_number):
                # explore next ghost, same depth
                next_ghost_index = index_of_ghost + 1
                final_val = self.random_ghost_action(next_gamestate, depth, next_ghost_index)

                # add to average_value
                average_value += (weight * final_val)

            else:
                # explore next layer --> agent
                # also divide into two situations
                # 2.1: reach bottom layer --> return score
                # 2.2: still in the mid layer --> return value from agent

                if(depth == self.depth):
                    bottom_val = self.evaluationFunction(next_gamestate)

                    # add to average_value
                    average_value += (weight * bottom_val)

                else:
                    final_val = self.agent_action(next_gamestate, depth+1)

                    # add to average_value
                    average_value += (weight * final_val)

        return average_value

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    '''
        useful functions:
        
        def getPacmanState(self):
        def getPacmanPosition(self):
        def getGhostStates(self):
        def getGhostState(self, agentIndex):
        def getGhostPosition(self, agentIndex):
        def getGhostPositions(self):
        def getNumAgents(self):
        def getScore(self):
        def getCapsules(self):
        def getNumFood(self):
    '''

    # the location of pacman
    GameState = currentGameState

    pacman_loc = GameState.getPacmanPosition()
    # the remained food (list)
    # number : len(food_list)
    food_list = GameState.getFood().asList()
    # the remained capsules
    # number : len(capsules)
    capsules = GameState.getCapsules()
    # current_score
    current_score = GameState.getScore()
    # the states of ghosts
    ghost_states = GameState.getGhostStates()

    # use manhattan_distance to evluate the distance
    def manhattan_distance(loc):
        x1, y1 = pacman_loc  # loc of pacman
        x2, y2 = loc         # loc of food
        return (abs(x2 - x1) + abs(y2 - y1))

    # the locs of ghosts (hunter and prey)
    prey_ghost = []
    hunter_ghost_closest_dis = 10000

    for ghost_state in ghost_states:
        loc = ghost_state.getPosition()
        distance = manhattan_distance(loc)
        scared_time = ghost_state.scaredTimer

        if(scared_time > (1 * distance)):
            # meaning it's possible to catch this ghost (prey)
            # higher value of (scared_time - distance) means
            # more possible to catch ghost
            prey_ghost.append(scared_time - distance)
        else:
            # meaning this ghost is still a hunter
            if(distance < hunter_ghost_closest_dis):
                hunter_ghost_closest_dis = distance

    shortest_food_dis = 0

    if (len(food_list) == 0):
        shortest_food_dis = 0
    else:
        food_locs = sorted(food_list, key=lambda loc: manhattan_distance(loc))
        shortest_food_dis = manhattan_distance(food_locs[0])

    # to evaluate a new state
    # we should consider three aspects of new state:
    # 1: food
    # -1.1: food number -- less <==> better
    # -1.2: food loc -- closer <==> better

    # 2: ghost
    # -2.1: hunter ghost -- further <==> better
    #       and when closest distance = 1, we must avoid this situation !!!
    # -2.2: prey ghost -- closer <==> better

    # 3: current_score -- higher <==> better


    # when shortest_ghost_dis <= 2, it is very dangerous
    # we mush avoid this situation!!!! so return -10000
    if (hunter_ghost_closest_dis <= 1):
        return -10000

    # not sure how to set these weights
    w1 = 1
    w2 = 15
    w3 = 2
    w4 = 30

    score = w1 * current_score + w2 * hunter_ghost_closest_dis \
            - w3 * shortest_food_dis

    for prey in prey_ghost:
        single_score = w4 * prey
        score += single_score

    return score



# Abbreviation
better = betterEvaluationFunction
