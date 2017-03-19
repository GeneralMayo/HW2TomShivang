import gym
import numpy as np
from PIL import Image
import time

import preprocessors
atari=preprocessors.AtariPreprocessor()
env=gym.make('SpaceInvaders-v0')
state=env.reset()
res=atari.process_state_for_network(state)
frames=preprocessors.HistoryPreprocessor()
for j in range (4):
	for i in range(4):
		img = Image.fromarray(frames.frames[i])
		img.show()
		time.sleep(0.5)
	(next_state, reward, is_terminal, info)=env.step(4)
	frames.process_state_for_network(next_state)
	