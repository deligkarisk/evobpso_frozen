import time

from evobpso.configuration.Configuration import Configuration


class OptimizationRunner:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def run(self):
        start_time = time.time()

        aggregated_history = []


        for i in range(0, self.configuration.iterations):
            print('starting iteration: ' + str(i))
            if i == 0:
                results = self.configuration.population.iterate(first_iter=True)
            else:
                results = self.configuration.population.iterate(first_iter=False)
            aggregated_history.append(results)

        end_time = time.time()
        elapsed_time = ((end_time - start_time) / 60) / 60
        print("Runtime: " + str(elapsed_time) + " hours.")

        results = {'optimization_elapsed_time': elapsed_time,
                   'optimization_results_history': aggregated_history,
                   'optimization_population': self.configuration.population}

        return results
