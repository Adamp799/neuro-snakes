import gym # type: ignore
import sys
sys.path.append("..")
import gym_snake
from gym_snake.envs.snake_env import SnakeEnv
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

# This script is for testing the Snake environment in the gym_snake package.

# Manually test the environment by controlling the snake with keyboard inputs.
# The user can input 'w', 'a', 's', 'd' to control the snake's movement.
def manual_test(env):
    env.reset()
    key = ""
    score = np.zeros(env.n_snakes)
    snakes_remaining = env.n_snakes
    snake = env.controller.snakes[0]
    while key != "q":
        env.render()
        print("Rewards: {} | Snakes Remaining: {}".format(score, snakes_remaining))
        actions = [0] * env.n_snakes
        for i in range(len(actions)):
            key = input("action " + str(i+1) + " : ")
            if key   == "w": actions[i] = 0
            elif key == "d": actions[i] = 1
            elif key == "s": actions[i] = 2
            elif key == "a": actions[i] = 3
            elif key == "q": break
            else: pass
        obs, reward, done, info = env.step(actions)
        score += np.array(reward)
        snakes_remaining = info["snakes_remaining"]
        if done: break;
    print("Rewards: {} | Snakes Remaining: {}".format(score, snakes_remaining));
    env.render(close=True)

# Test the environment with random actions
def random_test(env):
    env.reset()
    done = False
    score = np.zeros(env.n_snakes)
    snakes_remaining = env.n_snakes
    try:
        while not done:
            env.render()
            actions = [0] * env.n_snakes
            for i in range(len(actions)):
                actions[i] = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            score += np.array(reward)
            snakes_remaining = info["snakes_remaining"]
    except KeyboardInterrupt:
        print("Manual interrupt")
    finally:
        print("Rewards: {} | Snakes Remaining: {}".format(score, snakes_remaining))
        env.render(close=True)

# Function to create a custom Snake environment with specified parameters
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
