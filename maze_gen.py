import numpy as np

def generate_maze(n: int, m: int, p: float) -> np.ndarray:
	"""
	A method that generates an NxM maze with barriers at random locations.

	Args:
		n: The number of rows in the maze
		m: The number of columns in the maze
		p: The probability of a cell in the maze being a barrier

	Returns:
		A numpy ndarray of strings containing the characters
		'.' indicating an empty cell
		'-' indicating a barrier
		'S' indicating the start cell
		'G' indicating the goal cell
	"""
	EMPTY = '.'
	BARRIER = '-'
	START = 'S'
	GOAL = 'G'

	maze = np.ndarray((n, m), dtype="U1")
	starting_pos = (np.random.randint(n), np.random.randint(m))
	goal_pos = starting_pos
	while goal_pos == starting_pos:
		goal_pos = (np.random.randint(n), np.random.randint(m))

	for i in range(n):
		for j in range(m):
			if (i, j) == starting_pos:
				maze[i, j] = START
			elif (i, j) == goal_pos:
				maze[i, j] = GOAL
			elif np.random.uniform() < p:
				maze[i, j] = BARRIER
			else:
				maze[i, j] = EMPTY

	print(maze)
	return maze


if __name__ == '__main__':
	generate_maze(6, 6, 0.1)
