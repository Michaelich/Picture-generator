from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
from multiprocessing import Manager
import time
import numpy as np
import cv2
from multiprocessing import set_start_method


def worker_eval(osobnik, index, result):
    picture2 = np.ones((240, 161, 3), np.uint8) * 255  # White background

    for circle in osobnik:
        picture = picture2.copy()
        cv2.circle(picture, (int(circle[0] * 161), int(circle[1] * 240)), int(circle[2] * 35),
                   (int(circle[5] * 255), int(circle[4] * 255), int(circle[3] * 255)), -1)
        # Adding alpha
        picture2 = cv2.addWeighted(picture, circle[6], picture2, 1 - circle[6], 0)

    result[index] = -(ssim(picture2[:,:,0], image1[:,:,0])+ssim(picture2[:,:,1],image1[:,:,1])+ssim(picture2[:,:,2],image1[:,:,2]))/3

def worker_next_pop(osobnik, index, mutation_probability, how_many, alfa, circles, param, off_spring, off_spring_val):
    if np.random.random() < mutation_probability:
        osobnik[circles, param] += (np.random.rand(how_many) - 0.5) / alfa
        osobnik[circles, param] = np.clip(osobnik[circles, param], 0, 1)
    else:
        osobnik[circles, param] = np.random.rand(how_many)

    picture2 = np.ones((240, 161, 3), np.uint8) * 255  # White background

    for circle in osobnik:
        picture = picture2.copy()
        cv2.circle(picture, (int(circle[0] * 161), int(circle[1] * 240)), int(circle[2] * 35),
                   (int(circle[5] * 255), int(circle[4] * 255), int(circle[3] * 255)), -1)
        # Adding alpha
        picture2 = cv2.addWeighted(picture, circle[6], picture2, 1 - circle[6], 0)

    # Children Evaluation
    off_spring_val[index] = -(ssim(picture2[:, :, 0], image1[:, :, 0]) + ssim(picture2[:, :, 1], image1[:, :, 1]) + ssim(
        picture2[:, :, 2], image1[:, :, 2])) / 3

    off_spring.put((osobnik, index))
    # Cross
    # Mut
    # Eval
    return


def create_picture(chromosome, width, height):
    '''
    :param chromosome: made of circle
    :param circle: [x, y, radius, r, g, b, a]
    :param width: of picture
    :param height: of picture
    :return:
    '''
    picture2 = np.ones((height, width, 3), np.uint8) * 255 # White background
    for circle in chromosome:
        picture = picture2.copy()
        cv2.circle(picture, (int(circle[0]*width), int(circle[1]*height)), int(circle[2]*35), (int(circle[5]*255), int(circle[4]*255), int(circle[3]*255)), -1)
        # Adding alpha
        picture2 = cv2.addWeighted(picture, circle[6], picture2, 1 - circle[6], 0)
    return picture2


image1 = cv2.imread("mona_lisa.png")

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
    Child2 = ind1.copy()
    Child = ind2.copy()
    Child2[ymin:ymax, xmin:xmax] = ind2[ymin:ymax, xmin:xmax]
    Child[ymin:ymax, xmin:xmax] = ind1[ymin:ymax, xmin:xmax]
    return Child, Child2


if __name__ == '__main__':
    set_start_method('spawn')
    with Pool() as p:
        population_size = 1
        chromosome_length = 250
        number_of_offspring = 1
        crossover_probability = 0.05
        mutation_probability = 0.75
        number_of_iterations = 50010
        alfa = 2.75

        manager = Manager()

        pop_dict = manager.dict()
        off_spring_dict = manager.dict()
        off_spring_pop = manager.Queue()

        time0 = time.time()
        SGA_costs = np.zeros(number_of_iterations)

        best_chromosome = np.zeros((chromosome_length, 7))
        best_objective_value = np.inf
        # Generating an initial population
        current_population = np.zeros((population_size, chromosome_length, 7), dtype=np.float64)
        for i in range(population_size):
            current_population[i, :, :] = np.random.sample(size=(chromosome_length, 7))

        # evaluating the objective function on the current population
        res = [p.apply_async(worker_eval, args=(current_population[i, :], i, pop_dict)) for i in range(population_size)]

        for r in res:
            r.wait()

        objective_values = np.zeros(population_size)
        for i in range(population_size):
            objective_values[i] = pop_dict[i]
        photo_counter=0
        for t in range(number_of_iterations):
            print(t)

            # Parent Selection
            fitness_values = objective_values.max() - objective_values
            if fitness_values.sum() > 0:
                fitness_values = fitness_values / fitness_values.sum()
            else:
                fitness_values = np.ones(population_size) / population_size
            parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(
                np.int64)

            # Children Generation
            children_population = np.zeros((number_of_offspring, chromosome_length, 7), dtype=np.float64)
            for i in range(number_of_offspring):
                children_population[i, :, :] = current_population[parent_indices[i], :, :]

            # Mutation
            how_many = np.random.randint(0, chromosome_length)
            param = np.random.randint(0, 6)
            circles = np.random.choice(chromosome_length, how_many, replace=False)

            res = [p.apply_async(worker_next_pop, args=(
                    children_population[i, :], i, mutation_probability, how_many, alfa, circles, param,
                    off_spring_pop,
                    off_spring_dict)) for i in range(number_of_offspring)]
            for r in res:
                r.wait()



            children_objective_values = np.zeros(number_of_offspring)
            while not off_spring_pop.empty():
                osobnik, index = off_spring_pop.get()
                children_population[index, :, :] = osobnik
                children_objective_values[index] = off_spring_dict[index]

            # New population
            objective_values = np.hstack([objective_values, children_objective_values])
            current_population = np.vstack([current_population, children_population])
            I = np.argsort(objective_values)
            current_population = current_population[I[:population_size], :]
            objective_values = objective_values[I[:population_size]]
            if best_objective_value > objective_values[0]:
                best_objective_value = objective_values[0]
                best_chromosome = current_population[0, :, :]
            SGA_costs[t] = objective_values[0]

            # Visualization
            P = create_picture(best_chromosome, 161, 240)
            Horizontal = np.concatenate((P, image1), axis=1)
            cv2.imshow('Picture', Horizontal)
            if t % 2500 == 0:
                cv2.imwrite(f"resultMP{photo_counter}.jpg", P)
                photo_counter += 1
            cv2.waitKey(1)
        # return SGA_costs, best_chromosome


# Narysować lub wyprintować po otrzymaniu
# return SGA_costs, best_chromosome
