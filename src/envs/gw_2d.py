from .multiagentenv import MultiAgentEnv
from functools import reduce
import numpy as np

class GridWorld2D(MultiAgentEnv):

	def __init__(self, **kwargs):

		self.map_size = [15, 15] # minimum size
		self.obs_range = [5, 5]
		self.state = np.zeros(self.map_size)
		self.terrains = {
			0 : 'ground',
			1 : 'water',
			2 : 'switch',
			3 : 'bridge',
			4 : 'goal',
			5 : 'agent'
		}
		# no pass-through in water, but agent can see the opposite side.
		# bridge is invisible unless one agent is right on the switch.
		# agent can override other terrains (goal, switch, bridge).
		self.actions = {
			0 : 'step forward',
			1 : 'step right',
			2 : 'step backward',
			3 : 'step left',
			4 : 'stand still',
			5 : 'rotate left',
			6 : 'rotate right'
		}
		self.n_actions = 7
		# the orientation may affect the observation of one agent.
		# all actions are available throughout the game.
		self.orientations = {
			0 : 'north',
			1 : 'east',
			2 : 'south',
			3 : 'west'
		}
		self.n_agents = 2
		self.episode_limit = 100
		self.randomized = True
		self.reset()

	def reset(self):
		max_r = self.map_size[0]-1
		max_c = self.map_size[1]-1
		self.bridge_visible = False
		self.t_step = 0
		if self.randomized:
			self.water_column = np.random.random_integers(2, max_c-1)
			self.bridge_row = np.random.random_integers(0, max_c)
			self.goal_pos = [np.random.random_integers(0, max_r), np.random.random_integers(0, self.water_column-1)]
			self.agent_pos = [[np.random.random_integers(0, max_r), np.random.random_integers(self.water_column+1, max_c)] for i in range(self.n_agents)]
			self.switch_pos = [np.random.random_integers(0, max_r), np.random.random_integers(self.water_column+1, max_c)]
			self.agent_ori = [np.random.random_integers(0 ,4) for _ in range(self.n_agents)]
		else:
			self.water_column = 5
			self.bridge_row = 10
			self.goal_pos = [max_r-1, 0]
			self.switch_pos = [0, max_c-1]
			self.agent_pos = [[np.random.random_integers(0, max_r), np.random.random_integers(self.water_column+1, max_c)] for i in range(self.n_agents)]
			self.agent_ori = [np.random.random_integers(0 ,4) for _ in range(self.n_agents)]

		self._cal_state()
		return self.get_obs(), self.get_state()

	def _cal_state(self):
		state = np.zeros(self.map_size)
		state[:,self.water_column] = 1.
		if self.bridge_visible:
			state[self.bridge_row, self.water_column] = 3.
		state[tuple(self.goal_pos)] = 4.
		state[tuple(self.switch_pos)] = 2.
		for i in range(self.n_agents):
			state[tuple(self.agent_pos[i])] = 5.
		self.state = state
		return state

	def get_state(self):
		return self._cal_state()

	def get_state_size(self):
		return self.map_size

	def get_obs(self):
		agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
		return agents_obs

	def get_obs_agent(self, agent_id):
		obs = np.ones(self.obs_range)
		max_r = self.map_size[0]-1
		max_c = self.map_size[1]-1
		delta_h = 2
		delta_v = 4
		x = self.agent_pos[agent_id][0]
		y = self.agent_pos[agent_id][1]
		ori = self.agent_ori[agent_id]
		if ori == 0:
			row_l = max(0, x-delta_v)
			row_h = x+1
			col_l = max(0, y-delta_h)
			col_h = min(max_c, y+delta_h)+1
			offset_x = x-delta_v
			offset_y = y-delta_h
		elif ori == 1:
			row_l = max(0, x-delta_h)
			row_h = min(max_r, x+delta_h)+1
			col_l = y
			col_h = min(max_c, y+delta_v)+1
			offset_x = x-delta_h
			offset_y = y
		elif ori == 2:
			row_l = x
			row_h = min(max_r, x+delta_v)+1
			col_l = max(0, y-delta_h)
			col_h = min(max_c, y+delta_h)+1
			offset_x = x
			offset_y = y-delta_h
		else:
			row_l = max(0, x-delta_h)
			row_h = min(max_r, x+delta_h)+1
			col_l = max(0, y-delta_v)
			col_h = y+1
			offset_x = x-delta_h
			offset_y = y-delta_v
		obs[(row_l-offset_x):(row_h-offset_x), (col_l-offset_y):(col_h-offset_y)] = self.state[row_l:row_h, col_l:col_h]
		return obs

	def get_obs_size(self):
		return self.obs_range

	def _visualize(self, state):
		rep = str()
		state = self.state.astype(int)
		for i in range(self.map_size[0]):
			rep += (reduce(lambda x, y:str(x)+str(y) ,state[i,:])+'\n')	
		return rep

	def render(self):
		print(self._visualize(self.state))

	def _cal_pos(self, pos, ori, a):
		delta = {0 : [-1, 0],
			1 : [0, 1],
			2 : [1, 0],
			3 : [0, -1]
		}
		new_pos = [sum(x) for x in zip(pos, delta[(ori+a)%4])]
		return new_pos

	def _out_of_map(self, pos):
		if pos[0]<0 or pos[1]<0 or pos[0]>=self.map_size[0] or pos[1]>=self.map_size[1]:
			return True
		else:
			return False

	def step(self, actions):
		# assume actions are numbers, not one-hot reprs
		info = {}
		self.last_actions = actions
		if self.agent_pos[1] == self.switch_pos:
			self.bridge_visible = True
		else:
			self.bridge_visible = False
		for i in range(self.n_agents):
			if actions[i] == 4:
				continue
			elif actions[i] == 5:
				self.agent_ori[i] -= 1
				self.agent_ori[i] %= 4
			elif actions[i] == 6:
				self.agent_ori[i] += 1
				self.agent_ori[i] %= 4
			else:
				new_pos = self._cal_pos(self.agent_pos[i], self.agent_ori[i], actions[i])
				if self._out_of_map(new_pos):
					continue
				elif (new_pos[0] == self.bridge_row) and self.bridge_visible:
					self.agent_pos[i] = new_pos
				elif new_pos[1] != self.water_column:
					self.agent_pos[i] = new_pos
				else:
					continue

		if self.agent_pos[0] == self.goal_pos:
			reward = 1
			terminal = True
		else:
			reward = 0
			terminal = False

		self.t_step += 1
		if self.t_step >= self.episode_limit:
			terminal = True

		self._cal_state()
		return reward, terminal, info 

	def get_total_actions(self):
		return self.n_actions

	def get_avail_agent_actions(self, agent_id):
		return [1]*self.n_actions

	def get_avail_actions(self):
		return [1]*self.n_actions*self.n_agents

	def close(self):
		pass

	def seed(self):
		pass

	def save_replay(self):
		pass