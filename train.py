from maze_env import MazeEnv
from dynamic_programming import policy_iteration, value_iteration
from util import print_policy

if __name__ == '__main__':
    env = MazeEnv(10, 10, 0.3)

    with open('mazefile', 'w') as f:
        f.write(str(env.maze))

    # obtain the optimal policy and optimal state-value function
    print('\n')
    LINELEN = 100
    print('\t\tValue Iteration')
    print('-' * LINELEN)
    policy_pi, V_pi = value_iteration(env)

    # print the optimal policy
    print("Optimal Policy:", '\n')
    print_policy(policy_pi, env.nrow, env.ncol, env.maze)
    print('-' * LINELEN)

    print('\n')

    print('\t\tPolicy Iteration')
    print('-' * LINELEN)

    policy_pi, V_pi = truncated_policy_iteration(env)

    # print the optimal policy
    print("Optimal Policy:", '\n')
    print_policy(policy_pi, env.nrow, env.ncol, env.maze)
    print('-' * LINELEN)
