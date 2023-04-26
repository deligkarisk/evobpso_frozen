from unittest.mock import Mock

from evobpso.population.Population import Population
from evobpso.scheme.Scheme import Scheme

params = Mock()
scheme = Scheme(version='boolean', variable_length=True, vmax=False, vmut=True, optimization_params=params)
scheme.compile(architecture_properties=Mock(), data_loader=Mock(), results_folder=Mock())
population = Population(scheme)  # adds the rest of the necessary properties
runner.run(population)
