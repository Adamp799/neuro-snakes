import gym
import sys
sys.path.append("..")
import gym_snake

print("q to quit, wasd to move")
env = gym.make('snake-v0')

done = False
reward = 0
obs = env.reset()
key = ""
action = 0
while key != "q":
    env.render()
    key = input("action: ")
    if key   == "w": action = 0
    elif key == "d": action = 1
    elif key == "s": action = 2
    elif key == "a": action = 3
    else: pass
    obs, reward, done, info = env.step(action)
    print("reward:", reward)
    print("done:", done)
    print("info --")
    for k in info.keys():
        print("  ", k, ":", info[k])
    if done:
        obs = env.reset()
env.render(close=True)
