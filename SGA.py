import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

def objective_function(p):
    P = create_picture(p, 125, 186)
    PR = cv2.cvtColor(np.float32(P), cv2.COLOR_BGR2GRAY)
    return mse(PR, image1)

import cv2
import numpy as np
from pandas.core.tools.datetimes import Scalar


def create_picture(chromosome, width, height):
    '''
    :param chromosome: made of circle
    :param circle: [x, y, radius, r, g, b]
    :param width: of picture
    :param height: of picture
    :return:
    '''
    picture = np.ones((height, width, 3), np.uint8)*255 # White background

    for cirlce in chromosome:
        cv2.circle(picture, (cirlce[0], cirlce[1]), cirlce[2], (int(cirlce[3]), int(cirlce[4]), int(cirlce[5])), -1)

    return picture

def Crossover(ind1, ind2):
    y1,x1 = np.random.randint(ind1.shape)
    y2,x2= np.random.randint(ind1.shape)
    xmin=min(x1,x2)
    xmax=max(x1,x2)
    ymin=min(y1,y2)
    ymax=max(y1,y2)
    Child2=ind1.copy()
    Child=ind2.copy()
    Child2[ymin:ymax, xmin:xmax] = ind2[ymin:ymax,xmin:xmax]
    Child[ymin:ymax, xmin:xmax] = ind1[ymin:ymax,xmin:xmax]
    return Child, Child2

def random_transposition_mutation(p):
    a = np.random.choice(len(p), 2, False)
    i, j = a.min(), a.max()
    q = p.copy()
    temp = q[i].copy()
    q[i] = q[j]
    q[j] = temp
    return q

def mse(img1, img2):
    height, width = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    return err/(float(height*width))

#insert image source
image1 = cv2.imread("mona_lisa.png")
#print(image1)
image1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_BGR2GRAY)

def SGA(population_size,chromosome_length,number_of_offspring, crossover_probability, mutation_probability, number_of_iterations):
  time0 = time.time()
  SGA_costs = np.zeros(number_of_iterations)

  best_objective_value =  np.Inf
  best_chromosome = np.zeros((chromosome_length, 6))
  # generating an initial population
  current_population = np.zeros([population_size, chromosome_length, 6], dtype=int)
  for i in range(population_size):
      current_population[i, :, :] = np.random.randint([0,0,0,0,0,0],[100,200,40,255,255,255], size=(chromosome_length,6))
  # evaluating the objective function on the current population
  objective_values = np.zeros(population_size)
  for i in range(population_size):
      objective_values[i] = objective_function(current_population[i, :])

  for t in range(number_of_iterations):
      print(t)
      # selecting the parent indices by the roulette wheel method
      fitness_values = objective_values.max() - objective_values
      if fitness_values.sum() > 0:
          fitness_values = fitness_values / fitness_values.sum()
      else:
          fitness_values = np.ones(population_size) / population_size
      parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)

      # creating the children population
      children_population = np.zeros([population_size, chromosome_length, 6], dtype=int)
      for i in range(int(number_of_offspring/2)):
          if np.random.random() < crossover_probability:
              children_population[2*i, :, :], children_population[2*i+1, :, :] = Crossover(current_population[parent_indices[2*i], :, :].copy(), current_population[parent_indices[2*i+1], : , :].copy())
          else:
              children_population[2*i, :, :], children_population[2*i+1, :, :] = current_population[parent_indices[2*i], :, :].copy(), current_population[parent_indices[2*i+1]].copy()
      if np.mod(number_of_offspring, 2) == 1:
          children_population[-1, :] = current_population[parent_indices[-1], :]

      # mutating the children population
      for i in range(number_of_offspring):
          if np.random.random() < mutation_probability:
              children_population[i, :] = random_transposition_mutation(children_population[i, :])

      # evaluating the objective function on the children population
      children_objective_values = np.zeros(number_of_offspring)
      for i in range(number_of_offspring):
          children_objective_values[i] = objective_function(children_population[i, :])

      # replacing the current population by (Mu + Lambda) Replacement
      objective_values = np.hstack([objective_values, children_objective_values])
      current_population = np.vstack([current_population, children_population])

      I = np.argsort(objective_values)
      current_population = current_population[I[:population_size], :]
      objective_values = objective_values[I[:population_size]]

      # recording some statistics
      if best_objective_value > objective_values[0]:
          best_objective_value = objective_values[0]
          best_chromosome = current_population[0, :]
      SGA_costs[t]=objective_values[0]
      #print('%3d %14.8f %12.8f %12.8f %12.8f %12.8f' % (t, time.time() - time0, objective_values.min(), objective_values.mean(), objective_values.max(), objective_values.std()))
  return SGA_costs, best_chromosome


number_of_circles = 75
number_of_iterations = 1000
number_of_population = 1000
number_of_offspring = 1000
SA_costs, best_chromo = SGA(number_of_population, number_of_circles, number_of_offspring, 0.95, 0.05, number_of_iterations)
print(SA_costs)
cv2.imshow('Picture',create_picture(best_chromo, 125, 186))

cv2.waitKey(0)

plt.figure(figsize=(12,8))
plt.plot(SA_costs)
plt.show()