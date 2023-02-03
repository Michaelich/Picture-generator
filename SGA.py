from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

r=90
def objective_function(p):
    P = create_picture(p, 161, 240)
    return compare_color(P, image1)

def create_picture(chromosome, width, height):
    '''
    :param chromosome: made of circle
    :param circle: [x, y, radius, r, g, b, a]
    :param width: of picture
    :param height: of picture
    :return:
    '''
    picture2 = np.ones((height, width, 3), np.uint8)*255 # White background
    for circle in chromosome:
        picture = picture2.copy()
        #print(circle)
        cv2.circle(picture, (int(circle[0]*width), int(circle[1]*height)), int(circle[2]*r+10), (int(circle[5]*255), int(circle[4]*255), int(circle[3]*255)), -1)
        # Adding alpha
        picture2 = cv2.addWeighted(picture, circle[6], picture2, 1 - circle[6], 0)
    return picture2

def Crossover(ind1, ind2):
    y1, x1 = np.random.randint(ind1.shape)
    y2, x2 = np.random.randint(ind1.shape)
    if x1 < x2:
        xmin, xmax = x1, x2
    else:
        xmin, xmax = x2, x1
    if y1 < y2:
        ymin, ymax = y1, y2
    else:
        ymin, ymax = y2, y1
    Child2=ind1.copy()
    Child=ind2.copy()
    Child2[ymin:ymax, xmin:xmax] = ind2[ymin:ymax, xmin:xmax]
    Child[ymin:ymax, xmin:xmax] = ind1[ymin:ymax, xmin:xmax]
    return Child, Child2

def compare_color(img1, img2):
    return -(ssim(img1[:,:,0], img2[:,:,0])+ssim(img1[:,:,1],img2[:,:,1])+ssim(img1[:,:,2],img2[:,:,2]))/3

image1 = cv2.imread("mona_lisa.png")

def SGA(population_size, chromosome_length, number_of_offspring, alfa, number_of_iterations, mutation_probability, sigma):
    # ES
    tau = 1/np.sqrt(2*chromosome_length)
    tau0 = 1/np.sqrt(2*np.sqrt(chromosome_length))
    current_population_sigmas = sigma * np.ones((population_size, chromosome_length, 7))
    add_circles_time = 0  # How many iterations that function didn't improve (goes back to 0 when circles are added)
    add_circles_expected_time = 50  # How many iterations without improve after we add circles (Increase after time)
    add_circles_epsilon = 0  # How small changes counts as not improvement (gets smaller every time we add circles)
    add_circles_how_many = 1  # How many circles add in one go
    max_chromosome_length = 350  # Max number of circles that can be added overtime
    SGA_costs = np.zeros(number_of_iterations)
    best_chromosome = np.zeros((chromosome_length, 7))
    best_objective_value = np.inf
    # Generating an initial population
    current_population = np.zeros((population_size, chromosome_length,7), dtype=np.float64)
    for i in range(population_size):
        current_population[i, :, :] = np.random.sample(size=(chromosome_length,7))
    # Evaluating the objective function on the current population
    objective_values = np.zeros(population_size)
    for i in range(population_size):
        objective_values[i] = objective_function(current_population[i,:,:])
    # Main loop
    for t in range(number_of_iterations):
        print(t)

        # Parent Selection
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(population_size) / population_size
        parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)

        # Children Generation
        children_population_sigmas = np.zeros((number_of_offspring, chromosome_length, 7))
        children_population = np.zeros((number_of_offspring, chromosome_length, 7), dtype=np.float64)
        for i in range(number_of_offspring):
            children_population[i,:,:] = current_population[parent_indices[i],:,:]
            children_population_sigmas[i, :,:] = current_population_sigmas[parent_indices[i], :, :]
        # Crossover
        """ For now we don't do that"""

        # Mutation
        param = np.random.randint(0, 7)
        if np.random.random() < 0.10:
            how_many = 1
            circles = np.array([chromosome_length - 1])

        else:
            how_many = np.random.randint(1, chromosome_length+1)
            circles = np.random.choice(chromosome_length, how_many, replace=False)
        #print(circles)
        children_population_sigmas = children_population_sigmas * np.exp(
            tau * np.random.randn(number_of_offspring, how_many, 7) + tau0 * np.random.randn(number_of_offspring, 1))
        for i in range(number_of_offspring):
            children_population[i, circles, param] += children_population_sigmas[i,circles,param] * np.random.randn(how_many)
            children_population[i, circles, param] = np.clip(children_population[i,circles, param], 0, 1)

        # Children Evaluation
        children_objective_values = np.zeros(number_of_offspring)
        for i in range(number_of_offspring):
            children_objective_values[i] = objective_function(children_population[i])

        # New population
        objective_values = np.hstack([objective_values, children_objective_values])
        current_population = np.vstack([current_population, children_population])
        current_population_sigmas = np.vstack([current_population_sigmas, children_population_sigmas])
        I = np.argsort(objective_values)
        current_population_sigmas = current_population_sigmas[I[:population_size], :]
        current_population = current_population[I[:population_size], :]
        objective_values = objective_values[I[:population_size]]
        if best_objective_value > objective_values[0]:
            if chromosome_length != max_chromosome_length:
                add_circles_time = 0
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :, :]
        elif chromosome_length != max_chromosome_length:
            add_circles_time += 1

        if add_circles_time >= add_circles_expected_time and chromosome_length + add_circles_how_many <= max_chromosome_length:
            print("ADDED CIRCLE")
            current_population = np.hstack(
                [current_population, np.random.sample(size=(population_size, add_circles_how_many, 7))])
            for i in range(population_size):
                objective_values[i] = objective_function(current_population[i, :, :])
            chromosome_length += add_circles_how_many
            print(f'Number of circles: {chromosome_length}')
            current_population_sigmas = np.hstack(
                [current_population_sigmas, sigma * np.ones((population_size, add_circles_how_many, 7))])
            I = np.argsort(objective_values)
            current_population = current_population[I[:population_size], :]
            objective_values = objective_values[I[:population_size]]
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :, :]
            tau = 1 / np.sqrt(2 * chromosome_length)
            tau0 = 1 / np.sqrt(2 * np.sqrt(chromosome_length))
            # Changing add circle variables
            add_circles_time = 0
            # add_circles_how_many += 1
            add_circles_epsilon *= 0.95
            add_circles_expected_time += 5
        elif add_circles_time >= add_circles_expected_time and chromosome_length != max_chromosome_length:
            print("ADDED CIRCLE")
            current_population = np.hstack(
                [np.random.sample(size=(population_size, add_circles_how_many, 7)), current_population])
            chromosome_length = max_chromosome_length
            print(f'Number of circles: {chromosome_length}')
            I = np.argsort(objective_values)
            current_population = current_population[I[:population_size], :]
            objective_values = objective_values[I[:population_size]]
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :, :]
            tau = 1 / np.sqrt(2 * chromosome_length)
            tau0 = 1 / np.sqrt(2 * np.sqrt(chromosome_length))
            # Changing add circle variables
            add_circles_time = 0
            # add_circles_how_many += 1
            add_circles_epsilon *= 0.95
            add_circles_expected_time += 5
        # print(current_population)
        SGA_costs[t] = objective_values[0]

        # Visualization
        P = create_picture(best_chromosome, 161, 240)
        Horizontal = np.concatenate((P, image1), axis=1)
        cv2.imshow('Picture', Horizontal)
        cv2.waitKey(1)
    return SGA_costs, best_chromosome

population_size = 1
number_of_circles = 1
number_of_offspring = 1

alfa = 2.75
number_of_iterations = 1000000
mutation_probability = 0.75
SA_costs, best_chromo = SGA(population_size,number_of_circles,number_of_offspring,
                            alfa, number_of_iterations, mutation_probability, 0.5)
print(SA_costs)
Picture = create_picture(best_chromo, 161, 240)
Hori = np.concatenate((Picture, image1), axis=1)
cv2.imshow('Picture',Hori)
cv2.waitKey(0)

plt.figure(figsize=(12,8))
plt.plot(SA_costs)
plt.show()
