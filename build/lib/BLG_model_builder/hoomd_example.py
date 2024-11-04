import hoomd
import matplotlib
import numpy
import itertools
import math
import gsd.hoomd

#matplotlib inline
matplotlib.style.use('ggplot')
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
integrator = hoomd.md.Integrator(dt=0.005)

sigma = 1
epsilon = 1
r = numpy.linspace(0.95, 3, 500)
V_lj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

fig = matplotlib.figure.Figure(figsize=(5, 3.09))
ax = fig.add_subplot()
ax.plot(r, V_lj)
ax.set_xlabel('r')
ax.set_ylabel('V')
fig

cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5
integrator.forces.append(lj)

nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
)

integrator.methods.append(nvt)

m = 4
N_particles = 4 * m**3
spacing = 1.3
K = math.ceil(N_particles ** (1 / 3))
L = K * spacing
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))

frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position[0:N_particles]
frame.particles.typeid = [0] * N_particles
frame.configuration.box = [L, L, L, 0, 0, 0]

frame.particles.types = ['A']
with gsd.hoomd.open(name='lattice.gsd', mode='x') as f:
    f.append(frame)

simulation = hoomd.Simulation()
simulation.create_state_from_gsd(filename='lattice.gsd')

integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
)
integrator.methods.append(nvt)
simulation.operations.integrator = integrator
snapshot = simulation.state.get_snapshot()
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All()
)
simulation.operations.computes.append(thermodynamic_properties)
simulation.run(0)

print("KE = ",thermodynamic_properties.kinetic_energy)
print("PE = ",thermodynamic_properties.potential_energy)

simulation.run(10000)