import numpy as np


def print_policy(policy, n, m, maze):
	policy_printed = np.ndarray((n, m), dtype="U1")
	i, j = 0, 0
	LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
	for action_prop in policy:
		if maze[i, j] == '-':
			policy_printed[i, j] = '-'
		elif maze[i, j] == 'G':
			policy_printed[i, j] = 'G'
		elif (action_prop[LEFT] == 1):
			policy_printed[i, j] = '←'
		elif (action_prop[DOWN] == 1):
			policy_printed[i, j] = '↓'
		elif (action_prop[RIGHT] == 1):
			policy_printed[i, j] = '→'
		elif (action_prop[UP] == 1):
			policy_printed[i, j] = '↑'
		if j == m - 1:
			i += 1
		j = (j + 1) % m
	print(policy_printed)
	

def print_iter(iter, V, env, policy=None):
	n, m, maze = env.nrow, env.ncol, env.maze
	print("Iteration: " + str(iter), "\n")
	print("Values:\n")
	print(np.reshape(V, (n, m)))
	if policy is not None:
		print("\nPolicy:\n")
		print_policy(policy, n, m, maze)
	LINELEN = 100
	print('-' * LINELEN)