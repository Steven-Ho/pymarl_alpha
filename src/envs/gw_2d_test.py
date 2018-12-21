from gw_2d import GridWorld2D
import numpy as np 

env = GridWorld2D()
env.render()

n_actions = env.get_total_actions()
while(True):
	a1 = np.random.random_integers(0, n_actions-1)
	a2 = np.random.random_integers(0, n_actions-1)
	_, terminal, _ = env.step([a1, a2])
	env.render(t=1000)
	if terminal:
		break