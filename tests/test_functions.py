import gym # type: ignore
import sys
sys.path.append("..")
import gym_snake
from gym_snake.envs.snake_env import SnakeEnv
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

def manual_test(env):
    print("q to quit, wasd to move")
    key = ""
    snakes_remaining = env.n_snakes
    env.reset()

    while key != "q":
        env.render()
        actions = [0]*snakes_remaining
        for i in range(snakes_remaining):
            key = input("action " + str(i+1) + " : ")
            if key   == "w": actions[i] = 0
            elif key == "d": actions[i] = 1
            elif key == "s": actions[i] = 2
            elif key == "a": actions[i] = 3
            else: pass
            if key == "q": break
        obs, rewards, done, info = env.step(actions)
        snakes_remaining = info["snakes_remaining"]
        plt.title("Rewards: {} | Snakes Remaining: {}".format(rewards, snakes_remaining))
        if done:
            env.reset()
    env.render(close=True)

def render_test(env):
    env.reset()
    done = False
    reward = np.zeros(env.n_snakes)
    snakes_remaining = env.n_snakes
    while not done:
        env.render()
        actions = [0]*env.n_snakes
        for i in range(len(actions)):
            actions[i] = env.action_space.sample()
        obs, rewards, done, info = env.step(actions)
        snakes_remaining = info["snakes_remaining"]
        reward += np.array(rewards)
        plt.title("Rewards: {} | Snakes Remaining: {}".format(reward, snakes_remaining))
    env.render(close=True)

def create_env(grid_size, unit_size, unit_gap, snake_size, n_snakes, n_foods, random_init):
    class CustomSnakeEnv(SnakeEnv):
        metadata = {'render.modes': ['human']}
        def __init__(self, grid_size=grid_size, unit_size=unit_size, unit_gap=unit_gap, snake_size=snake_size, n_snakes=n_snakes, n_foods=n_foods, random_init=random_init):
            super().__init__(
                grid_size=grid_size,
                unit_size=unit_size,
                unit_gap=unit_gap,
                snake_size=snake_size,
                n_snakes=n_snakes,
                n_foods=n_foods,
                random_init=random_init)
    
    env = CustomSnakeEnv()
    register(id='custom-snake', entry_point=lambda:env,)
    return gym.make('custom-snake')
