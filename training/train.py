import gym
import os, pickle, sys, math
sys.path.append('..')
import gym_snake
import numpy as np
import neat
from gym_snake.envs.snake_env import SnakeEnv
from gym.envs.registration import register
import time
import visualize

# Get inputs for Agent1.1
def get_inputs_1(env, snake):
    head = snake.head
    grid = env.controller.grid
    grid.height = grid.grid_size[1]
    grid.width = grid.grid_size[0]

    is_food_up = 1 if grid.food[1] < head[1] else 0
    is_food_right = 1 if grid.food[0] > head[0] else 0
    is_food_down = 1 if grid.food[1] > head[1] else 0
    is_food_left = 1 if grid.food[0] < head[0] else 0

    up = snake.step(head, 0)
    safe_up = 0 if grid.check_death(up) or snake.direction == 2 else 1
    right = snake.step(head, 1)
    safe_right = 0 if grid.check_death(right) or snake.direction == 3 else 1
    down = snake.step(head, 2)
    safe_down = 0 if grid.check_death(down) or snake.direction == 0 else 1
    left = snake.step(head, 3)
    safe_left = 0 if grid.check_death(left) or snake.direction == 1 else 1

    length_norm = (len(snake.body) + 1) / (grid.width * grid.height)
    return is_food_up, is_food_right, is_food_down, is_food_left, safe_up, safe_right, safe_down, safe_left, length_norm

# Get inputs for Agent2
def get_inputs_2(env, snake):
    head = snake.head
    grid = env.controller.grid
    grid.height = grid.grid_size[1]
    grid.width = grid.grid_size[0]

    food_direct_up = 1 if grid.food[1] < head[1] and grid.food[0] == head[0] else 0
    food_direct_right = 1 if grid.food[0] > head[0] and grid.food[1] == head[1] else 0
    food_direct_down = 1 if grid.food[1] > head[1] and grid.food[0] == head[0] else 0
    food_direct_left = 1 if grid.food[0] < head[0] and grid.food[1] == head[1] else 0
    food_diagonal_up_right = 1 if grid.food[0] > head[0] and grid.food[1] < head[1] else 0
    food_diagonal_up_left = 1 if grid.food[0] < head[0] and grid.food[1] < head[1] else 0
    food_diagonal_down_right = 1 if grid.food[0] > head[0] and grid.food[1] > head[1] else 0
    food_diagonal_down_left = 1 if grid.food[0] < head[0] and grid.food[1] > head[1] else 0
    # is_food_up = 1 if grid.food[1] < head[1] else 0
    # is_food_right = 1 if grid.food[0] > head[0] else 0
    # is_food_down = 1 if grid.food[1] > head[1] else 0
    # is_food_left = 1 if grid.food[0] < head[0] else 0

    distance_up = (head[1] / (grid.height - 1)).item()
    distance_right = ((grid.width - head[0] - 1) / (grid.width - 1)).item()
    distance_down = ((grid.height - head[1] - 1) / (grid.height - 1)).item()
    distance_left = (head[0] / (grid.width - 1)).item()
    # up = snake.step(head, 0)
    # safe_up = 0 if grid.check_death(up) or snake.direction == 2 else 1
    # right = snake.step(head, 1)
    # safe_right = 0 if grid.check_death(right) or snake.direction == 3 else 1
    # down = snake.step(head, 2)
    # safe_down = 0 if grid.check_death(down) or snake.direction == 0 else 1
    # left = snake.step(head, 3)
    # safe_left = 0 if grid.check_death(left) or snake.direction == 1 else 1
   
    body = [tuple(seg) for seg in snake.body]
    is_tail_up = 1 if any(x == head[0] and y < head[1] for x, y in body) else 0
    is_tail_right = 1 if any(x > head[0] and y == head[1] for x, y in body) else 0
    is_tail_down = 1 if any(x == head[0] and y > head[1] for x, y in body) else 0
    is_tail_left = 1 if any(x < head[0] and y == head[1] for x, y in body) else 0

    moving_up = 1 if snake.direction == 0 else 0
    moving_right = 1 if snake.direction == 1 else 0
    moving_down = 1 if snake.direction == 2 else 0
    moving_left = 1 if snake.direction == 3 else 0

    length_norm = (len(snake.body) + 1) / (grid.width * grid.height)
    return food_direct_up, food_direct_right, food_direct_down, food_direct_left, food_diagonal_up_right, food_diagonal_up_left, food_diagonal_down_right, food_diagonal_down_left, distance_up, distance_right, distance_down, distance_left, is_tail_up, is_tail_right, is_tail_down, is_tail_left, moving_up, moving_right, moving_down, moving_left, length_norm

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
    loop_punishment = -0.6 # Punishment for entering the same square
    near_food_score = 1.0 # Reward for moving closer to food
    move_punishment = -0.0
    turn_punishment = -0.0

    for i in range(0, len(genomes), num_snakes):
        env.reset()
        fitness = 0; score = 0; foods = 0
        hunger = 24
        pastPoints = set()
        head = env.controller.snakes[0].head
        grid = env.controller.grid
        distance_to_food = math.sqrt((head[0] - grid.food[0])**2 + (head[1] - grid.food[1])**2)
        direction = env.controller.snakes[0].direction

        done = False
        while not done:
            actions = []
            for j in range(num_snakes):
                _, g = genomes[i+j]
                net = neat.nn.FeedForwardNetwork.create(g, config)
                inputs = get_inputs_1(env, env.controller.snakes[j])
                outputs = net.activate(inputs)
                actions.append(np.argmax(outputs))

            obs, reward, done, info = env.step(actions)
            if env.controller.snakes[0] is None or env.controller.dead_snakes[0] is not None:
                break

            score += reward
            factor = ((foods + 3) / 3) # Factor related to snake length
            fitness += move_punishment 
            hunger -= 1
            if hunger <= 0: break
            head = env.controller.snakes[0].head
            if (head[0], head[1]) in pastPoints:
                fitness += loop_punishment 
            pastPoints.add((head[0], head[1]))

            grid = env.controller.grid
            new_distance_to_food = math.sqrt((head[0] - grid.food[0])**2 + (head[1] - grid.food[1])**2)
            new_direction = env.controller.snakes[0].direction
            if new_direction != direction:
                if np.abs(direction-new_direction) == 2: # Attempting to move backwards
                    break
                fitness += turn_punishment 
                direction = new_direction

            if reward > 0: # If Food eaten
                pastPoints.clear()
                foods += 1
                hunger = min(24 * factor, 144) # Reset hunger, increase by length factor
                fitness += 2 * factor
            elif new_distance_to_food < distance_to_food:
                fitness += near_food_score 
            #elif new_distance_to_food > distance_to_food and foods < 9:
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
        scores_array.append((generation_number, score))
        #print(f"Generation {generation_number} \tGenome {i} \tFoods {foods} \tB.Foods {best_foods} \tFitness {g.fitness} \tB.Fitness {best_fitness} \tScore {score}")
        
    save_best_instance(best_instance) # Save best instances to best_instances.pickle
    if generation_number % 20 == 0:
        save_object(pop, 'population.dat')
        print('Exporting population')

def run(environ, generations, file=None):
    global env
    global pop
    global scores_array # Record scores by generation for plotting
    env = environ
    scores_array = []
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

    if file is not None:
        pop = load_object(file)
        print("Reading population from " + file)
    else: pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    #pop.add_reporter(neat.Checkpointer(20))

    winner = pop.run(eval_fitness, generations)
    print('\nBest genome:\n{!s}'.format(winner.fitness))

    # node_names = {
    #     -1: 'food_direct_up',
    #     -2: 'food_direct_right',
    #     -3: 'food_direct_down',
    #     -4: 'food_direct_left',
    #     -5: 'food_diagonal_up_right',
    #     -6: 'food_diagonal_up_left',
    #     -7: 'food_diagonal_down_right',
    #     -8: 'food_diagonal_down_left',
    #     -9: 'distance_up',
    #     -10: 'distance_right',
    #     -11: 'distance_down',
    #     -12: 'distance_left',
    #     -13: 'is_tail_up',
    #     -14: 'is_tail_right',
    #     -15: 'is_tail_down',
    #     -16: 'is_tail_left',
    #     -17: 'moving_up',
    #     -18: 'moving_right',
    #     -19: 'moving_down',
    #     -20: 'moving_left',
    #     -21: 'length_norm',
    #     0: 'up',
    #     1: 'right',
    #     2: 'down',
    #     3: 'left'
    # }
    node_names = {
        -1: 'is_food_up',
        -2: 'is_food_right',
        -3: 'is_food_down',
        -4: 'is_food_left',
        -5: 'safe_up',
        -6: 'safe_right',
        -7: 'safe_down',
        -8: 'safe_left',
        -9: 'length_norm',
        0: 'up',
        1: 'right',
        2: 'down',
        3: 'left'
    }
    #best_instances = load_object('best_instances.pickle')
    #best_genome = sorted(best_instances, key=lambda x: x['fitness'], reverse=True)[0]['genome']
    visualize.draw_net(config, winner, node_names=node_names, filename='winner_net.png')
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=True, view=True)
    visualize.plot_species(stats, view=True)
    visualize.plot_scores(scores_array, view=True)

# Render the environment and test the given instance
def render_test(env, instance, frame_speed=0.05, render=True):
    env.reset()
    score = 0
    snakes_remaining = env.n_snakes
    net = instance['net']
    done = False
    try:
        while not done:
            if render: env.render(frame_speed=frame_speed)
            actions = []
            snakes = len([x for x in env.controller.snakes if x is not None])
            if snakes == 0: break
            for i in range(snakes):
                inputs = get_inputs_1(env, env.controller.snakes[i])
                outputs = net.activate(inputs)
                actions.append(np.argmax(outputs))
        
            obs, reward, done, info = env.step(actions)
            score += reward
            snakes_remaining = info["snakes_remaining"]
    except KeyboardInterrupt:
        print("Manual interrupt")
    finally:
        print("Rewards: {} | Snakes Remaining: {}".format(score, snakes))
        if render: env.render(frame_speed=frame_speed, close=True)
        return score

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
