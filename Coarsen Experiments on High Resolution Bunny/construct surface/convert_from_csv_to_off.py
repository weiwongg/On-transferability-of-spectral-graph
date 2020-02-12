import meshio 
from numpy import genfromtxt
import numpy as np
coords = genfromtxt('coarsen_high_resolution_bunny_coordinates.csv', delimiter=' ')
points = coords
cells = {
    "triangle": np.array([
                             [0, 1, 2]
                             ])
}
mesh = meshio.Mesh(points, cells)
meshio.write("coarsen_high_resolution_bunny.off", mesh)
