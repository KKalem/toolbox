import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import geometry as G

#  spherical_points = []
#  cartesian_points = []
#  colors = []
#
#  r = 1
#  for theta in np.linspace(0, np.pi, 30):
    #  for phi in np.linspace(-np.pi, np.pi, 40):
        #  uvr = [theta, phi, r]
        #  xyz = G.uvr_to_xyz(uvr)
        #  spherical_points.append(uvr)
        #  cartesian_points.append(xyz)
#
        #  if G.vec_len( np.array([theta, phi])-[np.pi/2, 1] ) > 0.3:
            #  colors.append('g')
        #  else:
            #  colors.append('b')
#
#
#  spherical_points = np.array(spherical_points)
#  cartesian_points = np.array(cartesian_points)
#
#
#  fig = plt.figure()
#  plt.axis('equal')
#  ax = fig.add_subplot(111,projection='3d')
#  plt.xlabel('x')
#  plt.ylabel('y')
#
#  ax.scatter3D(cartesian_points[:,0], cartesian_points[:,1], cartesian_points[:,2], c=colors)
#
#  plt.figure()
#  plt.axis('equal')
#  plt.scatter(spherical_points[:,0], spherical_points[:,1], c=colors)
#  plt.xlabel('theta')
#  plt.ylabel('phi')


