import multiprocesing as mp
from multiprocessing import Pool
from multiprocessing import Manager

def worker_eval(osobnik, index, function, result):
    result[index] = function(osobnik)

def worker_next_pop(p1, p2, function, index1, index2, off_spring, off_spring_val):
    #Cross
    #Mut
    #Eval
    return



if __name__ == '__main__':
    with Pool() as p:
        population_size = 1000
        chromosome_length = 75
        number_of_offspring = 1000
        crossover_probability = 0.05
        mutation_probability = 0.95
        number_of_iterations = 1000

        pop_dict = Manager.dict()
        off_spring_dict = Manager.dict()
        off_spring_pop = Manager.Queue()

        time0 = time.time()
        SGA_costs = np.zeros(number_of_iterations)

        best_objective_value = np.Inf
        best_chromosome = np.zeros((chromosome_length, 6))
        # generating an initial population
        current_population = np.zeros([population_size, chromosome_length, 6], dtype=int)
        for i in range(population_size):
            current_population[i, :, :] = np.random.randint([0, 0, 0, 0, 0, 0], [100, 200, 40, 255, 255, 255],
                                                            size=(chromosome_length, 6))


        # evaluating the objective function on the current population
        for i in range(population_size):
            pool.apply_async(worker_eval, args=(current_population[i, :], i, objective_function, pop_dict))
        objective_values = np.zeros(population_size)
        for i in range(population_size):
            objective_values[i] = pop_dict[i]

        pool.close()
        pool.join()

        for t in range(number_of_iterations):
            print(t)
            # selecting the parent indices by the roulette wheel method
            fitness_values = objective_values.max() - objective_values
            if fitness_values.sum() > 0:
                fitness_values = fitness_values / fitness_values.sum()
            else:
                fitness_values = np.ones(population_size) / population_size
            parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(
                np.int64)

            # creating the children population
            children_population = np.zeros([population_size, chromosome_length, 6], dtype=int)
            for i in range(int(number_of_offspring / 2)):
                if np.random.random() < crossover_probability:
                    children_population[2 * i, :, :], children_population[2 * i + 1, :, :] = Crossover(
                        current_population[parent_indices[2 * i], :, :].copy(),
                        current_population[parent_indices[2 * i + 1], :, :].copy())
                else:
                    children_population[2 * i, :, :], children_population[2 * i + 1, :, :] = current_population[
                                                                                             parent_indices[2 * i], :,
                                                                                             :].copy(), \
                                                                                             current_population[
                                                                                                 parent_indices[
                                                                                                     2 * i + 1]].copy()
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
            SGA_costs[t] = objective_values[0]
            # print('%3d %14.8f %12.8f %12.8f %12.8f %12.8f' % (t, time.time() - time0, objective_values.min(), objective_values.mean(), objective_values.max(), objective_values.std()))




# Narysować lub wyprintować po otrzymaniu
# return SGA_costs, best_chromosome