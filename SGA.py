import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import objectiveFunctions

def objective_function(p):
    P = p.reshape((15,15,3))
    PR = cv2.cvtColor(np.float32(P), cv2.COLOR_BGR2GRAY)
    return mse(PR, image1)

def OX(ind1, ind2):
    l=len(ind1)
    T=np.random.choice(ind1,2)
    a,b=T.min(),T.max()
    Child=ind1.copy()
    Child2=ind2.copy()
    Child2[a:b+1]=ind1[a:b+1]
    Child[a:b+1]=ind2[a:b+1]
    index1=0
    for i in range(l):
      if ind2[i] in Child2[a:b+1]:
        continue
      if index1==a:
        index1=b+1
      Child2[index1]=ind2[i]
      index1+=1
      if index1==l:
        break

    index2=0
    for i in range(l):
      if ind1[i] in Child[a:b+1]:
        continue
      #print(index1, i)
      if index2==a:
        index2=b+1
      Child[index2]=ind1[i]
      index2+=1
      if index2==l:
        break
    return Child, Child2

def random_transposition_mutation(p):
    a = np.random.choice(len(p), 2, False)
    i, j = a.min(), a.max()
    q = p.copy()
    q[i],q[j] = q[j],q[i]
    return q

def mse(img1, img2):
    height, width = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    return err/(float(height*width))

#insert image source
image1 = cv2.imread("SRC")
print(image1)
image1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_BGR2GRAY)

def SGA(population_size,chromosome_length,number_of_offspring, crossover_probability, mutation_probability, number_of_iterations):
  time0 = time.time()
  SGA_costs = np.zeros(number_of_iterations)

  best_objective_value =  np.Inf
  best_chromosome = np.zeros((1, chromosome_length))

  # generating an initial population
  current_population = np.zeros((population_size, chromosome_length), dtype=np.int64)
  for i in range(population_size):
      current_population[i, :] = np.random.randint(0,250,chromosome_length)

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
      children_population = np.zeros((number_of_offspring, chromosome_length), dtype=np.int64)
      for i in range(int(number_of_offspring/2)):
          if np.random.random() < crossover_probability:
              children_population[2*i, :], children_population[2*i+1, :] = OX(current_population[parent_indices[2*i], :].copy(), current_population[parent_indices[2*i+1], :].copy())
          else:
              children_population[2*i, :], children_population[2*i+1, :] = current_population[parent_indices[2*i], :].copy(), current_population[parent_indices[2*i+1]].copy()
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



chromosome_len=3*15*15 #rgb*width*height
SA_costs, best_chromo = SGA(750, chromosome_len, 750, 0.95, 0.05, 1500)
img = best_chromo.reshape((15,15,3))
print(img)
cv2.imwrite('result.jpg', img)