from Particle import Particle
from Population import Population
from Rastrigin import Rastrigin

problem = Rastrigin(steps=2**32 - 1, dimensions=2)
parent_pop = Population(20, problem, 32)


for i in range(0,100):
    parent_pop.iterate()
    #print(parent_pop.global_best_result)
    print(parent_pop.particles[0].current_position, parent_pop.particles[0].current_result)
