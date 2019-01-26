import numpy as np


def print_policy(policy, n, m, maze):
	policy_printed = np.ndarray((n, m), dtype="U1")
	i, j = 0, 0
	ARROWS = '←↓→↑'
	DARROWS = '⇐⇓⇒⇑'
	for action_prop in policy:
		if maze[i, j] == '-':
			policy_printed[i, j] = '-'
		elif maze[i, j] == 'G':
			policy_printed[i, j] = 'G'
		else:
			action = np.argmax(action_prop)
			policy_printed[i, j] = ARROWS[action] if maze[i, j] != 'S' else DARROWS[action]
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