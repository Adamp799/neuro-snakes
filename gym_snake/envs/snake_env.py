import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete
from matplotlib.backends.backend_agg import FigureCanvasAgg 
from IPython.display import clear_output, display

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.im = None
        self.random_init = random_init

        self.action_space = spaces.Discrete(4)

        controller = Controller(
            self.grid_size, self.unit_size, self.unit_gap,
            self.snake_size, self.n_snakes, self.n_foods,
            random_init=self.random_init)
        grid = controller.grid
        self.observation_space = spaces.Box(
            low=np.min(grid.COLORS),
            high=np.max(grid.COLORS),
        )

    def step(self, action):
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    # Reset the environment to an initial state
    # This method should be called at the beginning of each episode
    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()

        self.fig, self.ax = plt.subplots()
        plt.close()
        self.ax.axis('off')
        self.im = self.ax.imshow(self.last_obs)
        return self.last_obs

    # Render the environment
    def render(self, close=False, frame_speed=.05):
        if close:
            plt.close()
            return

        self.im.set_data(self.last_obs)  # Update image data
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        clear_output(wait=True)
        display(self.fig)
        plt.pause(frame_speed)

    def seed(self, x):
        pass
