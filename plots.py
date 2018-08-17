#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-14

import pickle
import numpy as np
import time
import sys
import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #needed for '3d'
plt.ion()

# toolbox imports
import geometry as G
from rosview import RosSwarmView, RosPoseView, RosMarkerArrayView
from dynamic_point import VelocityPoint
from dynamic_point_sphere_swarm import DynamicPointSphereSwarm
import plotting as P
import physical_stuff as phys
from Quaternion import Quat

cages = [
'cage [12 conical][2 planes][Aug 15 15:54].pickle',
'cage [4 spherical][2 planes][Aug 15 12:04].pickle',
'cage [6 conical][2 planes][Aug 15 15:46].pickle',
'cage [6 spherical][2 planes][Aug 15 15:46].pickle',
'cage [8 conical][2 planes][Aug 15 15:53].pickle',
'cage [100 conical][2 planes][Aug 16 15:33].pickle',
'cage [4 spherical][2 planes][Aug 17 15:36].pickle',
'cage [12 spherical][2 planes][Aug 17 15:47].pickle'
]

if __name__ == '__main__':
    # load a saved cage spec
    cage_file = cages[-1]
    with open(cage_file, 'rb') as fin:
        caging_data = pickle.load(fin)

    # xyz, r
    sphere_center, sphere_rad = caging_data['caged_sphere']
    # [ (A,n) ...]
    planes = caging_data['obstacle_planes']
    # [ xyz ... ]
    pos = caging_data['sensor_positions']
    # [ (K, 2, 3) ... ]. cant be an array, K is variable over the list
    edges_over_time = caging_data['cage_edges_over_time']
    # [ xyz ... ]
    ori = caging_data['agent_orientations']
    # r
    sensor_range = caging_data['sensor_range']
    # [0, 1, 2 ...] cumsum'd
    elapsed_time = caging_data['elapsed_time']
    # 'conical' or 'spherical'
    sensor_shape = caging_data['sensor_shape']

    ########################################################################################
    # plot the edge lengths over time
    edge_lens = []
    for t,edges in zip(elapsed_time, edges_over_time):
        for edge in edges:
            if edge is not None:
                L = G.euclid_distance(edge[0], edge[1])
                edge_lens.append((t,L))
    edge_lens = np.array(edge_lens)

    # edge_lens might not have the same number of edges at every time step!
    # so we can not really make a 2d array out of it to plot
    # we can, however, scatter what we got
    plt.figure()
    plt.scatter(edge_lens[:,0], edge_lens[:,1], marker='.', alpha=0.3)
    plt.axhline(1.63, linestyle='--')
    plt.xlabel('Simulated time')
    plt.ylabel('Edge lengths of the cage')
    plt.ylim(0,3)
    plt.title('Cage edge lengths over time')

    ########################################################################################
    # plot the cage
    fig, ax = P.make_3d_fig()
    fig.set_tight_layout(tight=True)
    ax.grid(False)
    P.scatter3(ax, pos, color='g', marker='*', s=220)
    final_cage = edges_over_time[-1]
    for edge in final_cage:
        ax.plot(edge[:,0], edge[:,1], edge[:,2], color='b')

    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)
    plt.title('')


    ########################################################################################
    # plot a sphere, check each point for coverage
    nu, nv = 20,20

    uu, vv = np.meshgrid(np.linspace(0,np.pi,nu), np.linspace(-np.pi, np.pi, nv))
    colors = []
    xyzs = []
    sp_uvs = []
    sp_colors = []
    for i in range(nu):
        for j in range(nv):
            xyz = G.uvr_to_xyz((uu[i,j],vv[i,j],1))
            # project point to the planes to see if it is 'outside' any
            distance = None
            for A,n in planes:
                projection, distance = G.project_point_to_plane(xyz, A, n)
                if distance < 0:
                    xyz = projection
                    break

            xyzs.append(xyz)
            sp_uvs.append(G.xyz_to_uvr(xyz))

            # color the planes differently
            if distance is not None and distance<0:
                colors.append(0.5)
                sp_colors.append(0.5)
                continue

            dist = G.euclid_distance(xyz, pos)
            mindist = np.min(dist)
            if mindist < sensor_range:
                colors.append(1)
                sp_colors.append(1)
            else:
                colors.append(0)
                sp_colors.append(0)

    # normalize colors to 0-1
    colors = np.array(colors, dtype='float64')
    maxdist = np.max(colors)
    colors /= maxdist

    #  fig, ax = P.make_3d_fig()
    #  fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99)
    c = P.surface_tri(ax, xyzs, uu, vv, colors, cmap='seismic', linewidth=0, alpha=0.7)

    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)



