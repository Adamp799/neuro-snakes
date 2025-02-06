import gym
import os, pickle, math, sys
sys.path.append('..')
import gym_snake
import numpy as np
import neat
from neat import nn, population

global pop
rendering = True

def look_to(direction, head, env):
    grid = env.grid
    snake = env.controller.snakes[0]
    height = env.grid_size[1]*env.unit_size
    width = env.grid_size[0]*env.unit_size

    tail_found = False
    food_found = False
    
    dist_food = width
    dist_tail = width
    dist = 0

    while not grid.off_grid(head):
      if not tail_found and grid.tail_space(head):
        tail_found = True
        dist_tail = dist
      if not food_found and grid.food_space(head):
        food_found = True
        dist_food = dist
  
      dist += 1
      if snake is None:
        break
      head = snake.step(head, direction)
    return dist_food

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_best_instance(instance, filename='best_instances.pickle'):
    instances = []
    if os.path.isfile(filename):
       instances = load_object(filename)
    instances.append(instance)
    save_object(instances, filename)

def eval_fitness(genomes, config):
   best_instance = None
   best_fitness = 0
   best_foods = 0
   genome_number = 0
   generation_number = 0
   loop_punishment = 0.1
   near_food_score = 0.1
   done = False

   for genome_id, g in genomes:
      net = nn.FeedForwardNetwork.create(g, config) # create neural network
      score = 0.0
      hunger = 100 # 100 steps without eating and the snake dies

      env = gym.make('snake-v0')
      env.reset()
      if rendering:
        env.render() 
      head = env.controller.snakes[0].head
      dist_to_food = math.sqrt((head[0]-env.grid.food[0])**2 + (head[1]-env.grid.food[1])**2)

      count = 0
      error = 0
      pastPoints = set()
      foods = 0

      while not done:
        count += 1
        outputs = net.activate([look_to(0, head, env), look_to(1, head, env), look_to(2, head, env), look_to(3, head, env)]) 
        direction = outputs.index(max(outputs))

        obs, rew, done, info = env.step(direction)
        hunger -= 1
        if hunger <= 0:
          break
         
        if (head[0], head[1]) in pastPoints:
          score -= loop_punishment
        pastPoints.add((head[0], head[1]))

        if rew > 0:
          pastPoints = set()
          hunger += 100
          foods += 1
          score += rew
        else: 
          if abs(head[0]-env.grid.food[0] + head[1]-env.grid.food[1]) <= 1:
              score += near_food_score

        if rendering:
            env.render()
      
      if rendering:
         print("score: ", score)
         env.render()
      
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

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')
pop = population.Population(config)

if len(sys.argv) > 1: # load population from file
    pop = load_object(sys.argv[1])
    print("Reading popolation from " + sys.argv[1])

pop.run(eval_fitness, 10000) 
