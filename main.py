from Particle import Particle
from Population import Population
from Rastrigin import Rastrigin

problem = Rastrigin(steps=2**32 - 1)
parent_pop = Population()
particle = Particle(problem, 12, 32, parent_pop)

print("ok")
