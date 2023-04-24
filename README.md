# evobpso

## Overview
evobpso is a toolbox for automatically designing neural architectures for image classification
based on evolutionary/swarm optimization algorithms.
The current version of the toolbox is using Boolean Particle Swarm Optimization [1] in a global optimization setting.


## Requirements
See the associated setup.py file.
Briefly, the algorithm works with Tensorflow 2, although other libraries can be used as well.


## Tests
Unit tests are small tests aimed to run quickly before commiting.
Integration tests on the other hand, actually go through the full optimization process, so they may take a very long time.


[1]: Deligkaris, K. V., Zaharis, Z. D., Kampitaki, D. G., Goudos, S. K., Rekanos, I. T., &#38; Spasos, M. N. (2009). Thinned Planar Array Design Using Boolean PSO With Velocity Mutation. IEEE Transactions on Magnetics, 45(3), 1490–1493. https://doi.org/10.1109/TMAG.2009.2012687