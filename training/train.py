import gym
import os, pickle, math, sys
sys.path.append('..')
import gym_snake
import numpy as np
import neat
from neat import nn, population
import matplotlib.pyplot as plt

global pop
rendering = True

def look_to(env, direction):
    snake = env.controller.snakes[0]
    if snake is None:
       return 0
    head = snake.head
    grid = env.grid

    tail_found = False
    food_found = False
    
    dist_tail = env.grid_size[0] * env.unit_size
    dist_food = env.grid_size[0] * env.unit_size
    dist = 0

    while not grid.off_grid(head) and not snake is None and not (tail_found and food_found):
      if not tail_found and grid.tail_space(head):
        tail_found = True
        dist_tail = dist
      if not food_found and grid.food_space(head):
        food_found = True
        dist_food = dist
      dist += 1
      head = snake.step(head, direction)
  
    return dist_tail

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
   env = gym.make('snake-v0')
   genome_number = 0
   best_instance = None
   generation_number = 0
   best_fitness = 0
   best_foods = 0
   loop_punishment = 0.1
   near_food_score = 0.1

   for genome_id, g in genomes:
      net = nn.FeedForwardNetwork.create(g, config) 
      score = 0
      hunger = 100 # 100 steps without eating and the snake dies
      done = False
      env.reset()

      head = env.controller.snakes[0].head
      dist_to_food = abs((head[0]-env.grid.food[0]) + (head[1]-env.grid.food[1]))

      count = 0
      pastPoints = set()
      foods = 0

      while not done:
        count += 1
        outputs = net.activate([look_to(env, 0), look_to(env, 1), look_to(env, 2), look_to(env, 3)]) 
        direction = outputs.index(max(outputs))
        obs, reward, done, info = env.step(direction)

        hunger -= 1
        if hunger <= 0: break
         
        if (head[0], head[1]) in pastPoints:
          score -= loop_punishment
        pastPoints.add((head[0], head[1]))

        score += reward
        if reward > 0:
          pastPoints = set()
          hunger += 100
          foods += 1
        elif dist_to_food <= 1:
           score += near_food_score
      
      g.fitness = score 
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
      print(f"Generation {generation_number} \tGenome {genome_number} \tFoods {foods} \tBF {best_foods} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
      genome_number += 1
  
   save_best_instance(best_instance)
   generation_number += 1
   if generation_number % 20 == 0:
      save_object(pop, 'population.dat')
      print('Exporting population')

############################################################################################################

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')
pop = population.Population(config)

if len(sys.argv) > 1: # load population from file
    pop = load_object(sys.argv[1])
    print("Reading popolation from " + sys.argv[1])

pop.run(eval_fitness, 100) 

best_instances = load_object('best_instances.pickle')
best_instances = sorted(best_instances, key=lambda x: x['fitness'], reverse=True)
print("Best instances: ", best_instances[:5])

