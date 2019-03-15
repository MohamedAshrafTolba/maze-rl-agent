import numpy as np
from gym import utils
from gym.envs.toy_text import discrete

from maze_gen import generate_maze


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class MazeEnv(discrete.DiscreteEnv):
    """
    A discrete environment representing a maze where an agent starts at a
    certain location and tries to find a path to the goal location while avoiding
    collisions with barriers.
    The maze is described using a grid like the following

        S...
        .-.-
        -..-
        --.G

    'S' : starting cell
    '.' : empty cell
    '-' : barrier
    'G' : goal cell

    The episode ends when the agent reaches the goal.
    The agent receives a reward of 1 if it reaches the goal, and zero otherwise.
    """

    def __init__(self, nrow=4, ncol=4, barrier_prob=0.3):
        self.maze = generate_maze(nrow, ncol, barrier_prob)
        self.nrow, self.ncol = self.maze.shape

        nA = 4
        nS = self.nrow * self.ncol

        isd = np.array(self.maze == 'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for row in range(nrow):
            for col in range(ncol):
                state = self.__get_state(row, col)
                for action in range(nA):
                    li = P[state][action]
                    character = self.maze[row, col]
                    if character == 'G':
                        li.append((1.0, state, 0, True))
                    else:
                        new_row, new_col = self.__get_next_state(row, col, action)
                        new_state = self.__get_state(new_row, new_col)
                        new_character = self.maze[new_row, new_col]
                        done = False
                        reward = -1.0
                        li.append((1.0, new_state, reward, done))

        self.P = P
        super(MazeEnv, self).__init__(nS, nA, P, isd)

    def __get_state(self, row, col):
        return row * self.ncol + col

    def __get_next_state(self, row, col, action):
        if action == 0: # left
            new_col = max(col-1, 0)
            if self.maze[row, new_col] != '-':
                col = new_col
        elif action == 1: # down
            new_row = min(row+1, self.nrow-1)
            if self.maze[new_row, col] != '-':
                row = new_row
        elif action == 2: # right
            new_col = min(col+1, self.ncol-1)
            if self.maze[row, new_col] != '-':
                col = new_col
        elif action == 3: # up
            new_row = max(row-1, 0)
            if self.maze[new_row, col] != '-':
                row = new_row
        return (row, col)
