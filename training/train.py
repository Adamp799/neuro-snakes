import gym
import os, pickle, math, sys
sys.path.append('..')
import gym_snake
import numpy as np
import neat
from neat import nn, population
import matplotlib.pyplot as plt

global pop
global generation_number
rendering = True

def look_to(env, direction):
    snake = env.controller.snakes[0]
    if snake is None:
       return 0, 0
    head = snake.head
    grid = env.grid

    if direction == 0:
       foodDir = grid.food[1] < head[1]
    elif direction == 1:
       foodDir = grid.food[0] > head[0]
    elif direction == 2:
       foodDir = grid.food[1] > head[1]
    elif direction == 3:
       foodDir = grid.food[0] < head[0]
   
    safeDir = not grid.check_death(snake.step(head, direction))
    return int(safeDir), int(foodDir)

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
   global generation_number
   global pop

   env = gym.make('snake-v0')
   genome_number = 0
   best_instance = None
   best_fitness = 0
   best_foods = 0
   loop_punishment = 0.1
   survival_score = 0.1
   near_food_score = 0.1

   for genome_id, g in genomes:
      net = nn.FeedForwardNetwork.create(g, config) 
      fitness = 0
      score = 0;
      hunger = 250 # steps without eating and the snake dies
      done = False
      env.reset()

      count = 0
      pastPoints = set()
      foods = 0
      head = env.controller.snakes[0].head
      dist_to_food = abs(head[0]-env.grid.food[0]) + abs(head[1]-env.grid.food[1])

      while not done:
        count += 1
        
        inputs = []
        for i in range(4):
           looked = look_to(env, i)
           inputs.append(looked[0])
           inputs.append(looked[1])
        print(inputs)

        outputs = net.activate(inputs) 
        direction = outputs.index(max(outputs))
        print(direction, outputs)
        obs, reward, done, info = env.step(direction)

        fitness += reward
        score += reward
        if env.controller.snakes[0] is None:
           break

        hunger -= 1
        if hunger <= 0: 
           break
        fitness += survival_score

        head = env.controller.snakes[0].head
         
        if (head[0], head[1]) in pastPoints:
           fitness -= loop_punishment
        pastPoints.add((head[0], head[1]))
        
        new_dist_to_food = abs(head[0]-env.grid.food[0]) + abs(head[1]-env.grid.food[1])
        print(new_dist_to_food, dist_to_food)
        if reward > 0:
           pastPoints = set()
           hunger += 250
           foods += 1
        elif new_dist_to_food < dist_to_food:
           fitness += near_food_score
        else:
           fitness -= near_food_score
        dist_to_food = new_dist_to_food
      
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
      print(f"Generation {generation_number} \tGenome {genome_number} \tFoods {foods} \tBF {best_foods} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
      genome_number += 1
  
   save_best_instance(best_instance)
   generation_number += 1
   if generation_number % 20 == 0:
      save_object(pop, 'population.dat')
      print('Exporting population')

############################################################################################################

def run_pop(gens, file=None):
   global generation_number
   global pop
   generation_number = 0
   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config')
   pop = population.Population(config)

   if file is not None: # load population from file
      pop = load_object(file)
      generation_number = pop.generation + 1
      print("Reading population from " + file)

   pop.run(eval_fitness, gens) 

def test_best_instance():
   best_instances = load_object('best_instances.pickle')
   best_instances = sorted(best_instances, key=lambda x: x['fitness'], reverse=True)
   print("Best instances: ", best_instances[:5])
   input("Press enter to continue")

   best_net = best_instances[0]['net']
   env = gym.make('snake-v0')
   env.reset()
   done = False
   score = 0
   while not done:
    env.render()
    print(score)

    inputs = []
    for i in range(4):
        looked = look_to(env, i)
        inputs.append(looked[0])
        inputs.append(looked[1])
    
    outputs = best_net.activate(inputs) 
    direction = outputs.index(max(outputs))
    obs, reward, done, info = env.step(direction)
    score += reward
