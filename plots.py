#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-14

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

if __name__ == '__main__':
    ########################################################################################
    # PLOTTING
    ########################################################################################
    sensor_range = 1 if use_cones else 0.5
    sensor_pos = spikes._pos if use_cones else swarm._pos

    # time, agent, (x,y,z)
    traces = np.array(traces)
    # time, edge, [(x,y,z), (x,y,z)]
    edges_in_time = np.array(edges)
    # time, agent, (u,v,w)
    applied_forces = np.array(applied_forces)
    # dts
    elapsed_time = np.cumsum(elapsed_time)

    ########################################################################################
    # plot the edge lengths over time
    edge_lens = []
    for t,edges in zip(elapsed_tim, edges_in_time):
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
    P.scatter3(ax, swarm._pos)
    P.scatter3(ax, np.array((0,0,0)), color='g')


    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)
    plt.title('')




    ########################################################################################
    # plot a sphere, check each point for coverage
    nu, nv = 100,100

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

            dist = G.euclid_distance(xyz, sensor_pos)
            mindist = np.min(dist)
            if mindist<sensor_range:
                colors.append(1)
                sp_colors.append(1)
            else:
                colors.append(0)
                sp_colors.append(0)

    # normalize colors to 0-1
    colors = np.array(colors, dtype='float64')
    maxdist = np.max(colors)
    colors /= maxdist

    fig, ax = P.make_3d_fig()
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99)
    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)
    # move the agents a little outside the sphere so we can see them
    pts = G.xyz_to_uvr(swarm._pos)
    pts[:,2] += 0.05
    pts = G.uvr_to_xyz(pts)
    P.scatter3(ax, pts, c='r')
    c = P.surface_tri(ax, xyzs, uu, vv, colors, cmap='seismic', linewidth=0, alpha=0.7)

    #############################################
    # spherical coords plot
    #############################################
    sp_colors = np.array(sp_colors)
    sp_uvs = np.array(sp_uvs)[:,:2]
    plt.figure()
    plt.scatter(sp_uvs[:,0], sp_uvs[:,1], c=sp_colors)
    plt.title('uv-coords version')



