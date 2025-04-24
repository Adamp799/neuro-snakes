import gym
import os, pickle, sys, math
sys.path.append('..')
import gym_snake
import numpy as np
import neat
from gym_snake.envs.snake_env import SnakeEnv
from gym.envs.registration import register
import time

def get_inputs(env, snake):
    head = snake.head
    grid = env.controller.grid
    grid.height = grid.grid_size[1]
    grid.width = grid.grid_size[0]

    distance_up = head[1] / (grid.height - 1)
    distance_down = (grid.height - head[1] - 1) / (grid.height - 1)
    distance_left = head[0] / (grid.width - 1)
    distance_right = (grid.width - head[0] - 1) / (grid.width - 1)

    food_x = (grid.food[0] - head[0]) / (grid.width - 1)
    food_y = (grid.food[1] - head[1]) / (grid.height - 1)
   
    moving_up = 1 if snake.direction == 0 else 0
    moving_right = 1 if snake.direction == 1 else 0
    moving_down = 1 if snake.direction == 2 else 0
    moving_left = 1 if snake.direction == 3 else 0

    return distance_up, distance_down, distance_left, distance_right, food_x, food_y, moving_up, moving_right, moving_down, moving_left

def get_inputs_2(env, snake):
    head = snake.head
    grid = env.controller.grid
    grid.height = grid.grid_size[1]
    grid.width = grid.grid_size[0]

    is_food_up = 1 if grid.food[1] < head[1] else 0
    is_food_right = 1 if grid.food[0] > head[0] else 0
    is_food_down = 1 if grid.food[1] > head[1] else 0
    is_food_left = 1 if grid.food[0] < head[0] else 0

    #distance_up = head[1] / (grid.height - 1)
    #distance_right = (grid.width - head[0] - 1) / (grid.width - 1)
    #distance_down = (grid.height - head[1] - 1) / (grid.height - 1)
    #distance_left = head[0] / (grid.width - 1)

    up = snake.step(head, 0)
    safe_up = 0 if grid.check_death(up) or snake.direction == 2 else 1
    right = snake.step(head, 1)
    safe_right = 0 if grid.check_death(right) or snake.direction == 3 else 1
    down = snake.step(head, 2)
    safe_down = 0 if grid.check_death(down) or snake.direction == 0 else 1
    left = snake.step(head, 3)
    safe_left = 0 if grid.check_death(left) or snake.direction == 1 else 1
    return is_food_up, is_food_right, is_food_down, is_food_left, safe_up, safe_right, safe_down, safe_left

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as f: obj = pickle.load(f)
    return obj

def save_best_instance(instance, filename='best_instances.pickle'):
    instances = []
    if os.path.isfile(filename):
        instances = load_object(filename)
    instances.append(instance)
    save_object(instances, filename)

def eval_fitness(genomes, config):
    genome_ids = [gid for gid, _ in genomes]
    num_snakes = env.n_snakes
    best_fitness = 0; best_foods = 0
    best_instance = None
    generation_number = pop.generation
    loop_punishment = -0.6
    near_food_score = 1.0
    move_punishment = -0.0
    for i in range(0, len(genomes), num_snakes):
        env.reset()
        fitness = 0; score = 0; foods = 0
        hunger = 50
        pastPoints = set()
        head = env.controller.snakes[0].head
        grid = env.controller.grid
        distance_to_food = math.sqrt((head[0] - grid.food[0])**2 + (head[1] - grid.food[1])**2)

        done = False
        while not done:
            actions = []
            for j in range(num_snakes):
                _, g = genomes[i+j]
                net = neat.nn.FeedForwardNetwork.create(g, config)
                inputs = get_inputs_2(env, env.controller.snakes[j])
                outputs = net.activate(inputs)
                actions.append(np.argmax(outputs))

            obs, reward, done, info = env.step(actions)
            if env.controller.snakes[0] is None or env.controller.dead_snakes[0] is not None:
                break
            score += reward
            factor = (foods + 3) / 3
            fitness += move_punishment * factor
            hunger -= 1
            if hunger <= 0: break
            head = env.controller.snakes[0].head
            if (head[0], head[1]) in pastPoints:
                fitness += loop_punishment * factor
            pastPoints.add((head[0], head[1]))

            grid = env.controller.grid
            new_distance_to_food = math.sqrt((head[0] - grid.food[0])**2 + (head[1] - grid.food[1])**2)
            if reward > 0:
                pastPoints.clear()
                foods += 1
                hunger += 50
                fitness += 3 * factor
            elif new_distance_to_food < distance_to_food:
                fitness += near_food_score * factor
            #elif new_distance_to_food > distance_to_food:
                #fitness -= near_food_score * factor
            distance_to_food = new_distance_to_food

        g.fitness = fitness 
        if best_instance is None or g.fitness > best_fitness:
            best_instance = {
                'genome': g,
                'score': score,
                'net': net,
                'fitness': g.fitness,
                'num_generation': generation_number,
            }
        best_foods = max(best_foods, foods)
        best_fitness = max(best_fitness, g.fitness)
        print(f"Generation {generation_number} \tGenome {i} \tFoods {foods} \tB.Foods {best_foods} \tFitness {g.fitness} \tB.Fitness {best_fitness} \tScore {score}")
        
    save_best_instance(best_instance)
    if generation_number % 20 == 0:
        save_object(pop, 'population.dat')
        print('Exporting population')

def run(environ, generations, file=None):
    global env, pop
    env = environ
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

    if file is not None:
        pop = load_object(file)
        print("Reading population from " + file)
    else: pop = neat.Population(config)

    pop.run(eval_fitness, generations)

def render_test(env, instance):
    env.reset()
    score = np.zeros(env.n_snakes)
    snakes_remaining = env.n_snakes
    net = instance['net']

    done = False
    while not done:
        env.render()
        actions = []
        snakes = len([x for x in env.controller.snakes if x is not None])
        if snakes == 0: break
        for i in range(snakes):
            inputs = get_inputs_2(env, env.controller.snakes[i])
            outputs = net.activate(inputs)
            actions.append(np.argmax(outputs))
    
        obs, reward, done, info = env.step(actions)
        score += np.array(reward)
        snakes_remaining = info["snakes_remaining"]
    print("Rewards: {} | Snakes Remaining: {}".format(score, snakes))
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