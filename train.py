from maze_env import MazeEnv
from dynamic_programming import policy_iteration

if __name__ == '__main__':
    env = MazeEnv(6, 6, 0.1)

    # print the state space and action space
    print(env.observation_space)
    print(env.action_space)

    # print the total number of states and actions
    print(env.nS)
    print(env.nA)

    # obtain the optimal policy and optimal state-value function
    policy_pi, V_pi = policy_iteration(env)

    # print the optimal policy
    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(policy_pi,"\n")
