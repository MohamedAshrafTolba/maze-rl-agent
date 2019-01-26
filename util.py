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

def print_path(policy, n, m, maze):
	start_pos = np.where(maze == 'S')
	end_pos = np.where(maze == 'G')
	
	path = ''
	x, y = start_pos
	ARROWS = '←↓→↑'
	delta = [(0, -1), (1, 0), (0, 1), (-1, 0)]
	i, max_path = 0, n * m

	while (x, y) != end_pos:
		action = np.argmax(policy[x * n + y])
		path += ARROWS[action]
		delta_x, delta_y = delta[action]
		x += delta_x
		y += delta_y
		i += 1
		if i == max_path or maze[x, y] == '-':
			print("Could not find a path from source to goal")
			return


	print("Path:")
	print(path)
	print("Cost: " + str(len(path)))