from maze_env import MazeEnv
from dynamic_programming import policy_iteration, value_iteration, truncated_policy_iteration
from util import print_policy, print_path
import time

if __name__ == '__main__':
    env = MazeEnv(10, 10, 0.3)

    with open('mazefile', 'w') as f:
        f.write(str(env.maze))

    # obtain the optimal policy and optimal state-value function
    print('\n')
    LINELEN = 100
    print('\t\tValue Iteration')
    print('-' * LINELEN)

    start_time = time.time()
    policy_pi, V_pi = value_iteration(env, max_iter=100)
    end_time = time.time()

    # print the optimal policy
    print("Optimal Policy:", '\n')
    print_policy(policy_pi, env.nrow, env.ncol, env.maze)
    print_path(policy_pi, env.nrow, env.ncol, env.maze)
    print("Runtime: " + str(end_time - start_time))
    print('-' * LINELEN)

    print('\n')

    print('\t\tPolicy Iteration')
    print('-' * LINELEN)

    start_time = time.time()
    policy_pi, V_pi = truncated_policy_iteration(env, max_it=100)
    end_time = time.time()

    # print the optimal policy
    print("Optimal Policy:", '\n')
    print_policy(policy_pi, env.nrow, env.ncol, env.maze)
    print_path(policy_pi, env.nrow, env.ncol, env.maze)
    print("Runtime: " + str(end_time - start_time))
    print('-' * LINELEN)
