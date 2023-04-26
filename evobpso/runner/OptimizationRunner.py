import time

from evobpso.population.Population import Population


class OptimizationRunner:

    def __init__(self, population: Population):
        self.population = population

    def run(self):
        start_time = time.time()

        aggregated_history = []


        for i in range(0, self.population.scheme.optimization_params.pso_params.iters):
            print('starting iteration: ' + str(i))
            if i == 0:
                results = self.population.iterate(first_iter=True)
            else:
                results = self.population.iterate(first_iter=False)
            aggregated_history.append(results)

        end_time = time.time()
        elapsed_time = ((end_time - start_time) / 60) / 60
        print("Runtime: " + str(elapsed_time) + " hours.")

        results = {'optimization_elapsed_time': elapsed_time,
                   'optimization_results_history': aggregated_history,
                   'optimization_population': self.population}

        return results
