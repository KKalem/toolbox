#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-07-26


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

if __name__=='__main__':
    ########################################################################################

    # N, num of agents
    N = 4
    # dt, time step per sim tick
    # during run, the simulation will change between these when needed
    dt_steps = [0.005, 0.01, 0.05]
    current_dt_step = 0
    # ups, updates per second for ros stuff to update
    ups = 60
    # ticker per view, how many sim ticks to run per view update
    ticks_per_view = 10

    # set to false when not running a profiler
    # this changes the ros init_node disable signal and plotting stuff
    profiling = False

    # init the usual ros stuff
    rospy.init_node('rosviewtest', anonymous=True, disable_signals=profiling)

    rate = rospy.Rate(ups)

    # create the swarm with random starting positions
    init_pos = (np.random.random((N,3))*2)-1
    centers = [0,0,0]
    radii = 1
    # with some small mass and damping
    swarm = DynamicPointSphereSwarm(init_pos=init_pos,
                                    center=centers,
                                    radius=radii,
                                    mass=0.01,
                                    damping=0.05)

    spike_init_pos = (np.random.random((N,3))*2)-1
    spikes = DynamicPointSphereSwarm(init_pos=spike_init_pos,
                                     center=init_pos,
                                     radius=[0.4]*N,
                                     mass=0.01,
                                     damping=0.05)

    # we want fancy stuff to be shown, put them in a marker array
    # make the meshes slightly transparent
    marker_array = RosMarkerArrayView()

    # where the mesh files to display are stored for viewing
    mesh_dir = '/home/ozer/Dropbox/projects/toolbox/'

    ########################################################################################
    # SWARM INIT
    ########################################################################################
    # create the swarm view so we can see stuff in rviz
    swarm_view = RosSwarmView(swarm=swarm,
                              mesh_path=mesh_dir+'known_arrow.dae',
                              mesh_rgba=(0.2, 0.8, 0.2, 1),
                              mesh_scale=(1, 1, 1))

    spike_view = RosSwarmView(swarm=spikes,
                              mesh_path=mesh_dir+'known_arrow.dae',
                              mesh_rgba=(0.8, 0.8, 0.2, 1),
                              mesh_scale=(1, 1, 1),
                              last_used_id=swarm_view.last_used_id+10)

    # we also want to see a plane and sphere in rviz, representing the cage and obstacle
    # create views for them. Since these are stationary, we wont be updating them much
    # we just need the position of this, it wont be moving
    for center, radius in zip(swarm._center, swarm._radius):
        sphere_body = VelocityPoint(init_pos=center)
        sphere_view = RosPoseView(body=sphere_body,
                                  topic='/sphere_cage')
        marker_array.add_view(sphere_view, mesh_dir+'known_sphere.dae',
                              (radius,radius,radius),
                              (0.2,0.2,0.8,0.2))

    ########################################################################################
    # PLANES
    ########################################################################################
    # points on planes
    plane_uvrs = [ (0,0,0.8), (0,0,-0.8), (2,1,0.6)]#, (1.2, 1.2, -0.6)]
    plane_views = []
    plane_bodies = []
    planes = []
    for uvr in plane_uvrs:
        # a plane that is a little inside the sphere at 0,0,0 with r=1
        #  plane_pos = G.uvr_to_xyz((np.pi/6, np.pi/4, 0.5))
        plane_pos = G.uvr_to_xyz(uvr)
        # the plane normal should be towards the sphere center at 0,0,0
        plane_normal = np.zeros_like(plane_pos) - plane_pos
        _, plane_normal = G.vec_normalize(plane_normal)

        # these points use the velocty for orientation
        plane_body = VelocityPoint(init_pos=plane_pos,
                                   init_vel=plane_normal)
        plane_view = RosPoseView(body=plane_body,
                                 topic='/plane_obstacle')
        plane_views.append(plane_view)
        plane_bodies.append(plane_body)
        marker_array.add_view(plane_view, mesh_dir+'known_plane.dae',(5,5,5),(0.8,0.2,0.2,0.5))
        planes.append((plane_pos, plane_normal))

    ########################################################################################
    # INITIALIZE
    ########################################################################################
    # before we update, initialize the marker array
    marker_array.init()

    # update these for a little while until rviz realizes whats up and shows them
    rate2 = rospy.Rate(200)
    for i in range(100):
        # and show the markers
        for plane_view in plane_views:
            plane_view.update()
        sphere_view.update()
        marker_array.update()
        rate2.sleep()

    # to be plotted later
    sphere_err = []
    traces = []
    edges = []
    applied_forces = []
    spike_applied_forces = []

    tick_times = []

    ########################################################################################
    # MAIN LOOP
    ########################################################################################
    # main sim loop
    while not rospy.is_shutdown():
        # update the physics of the swarm many times per view update
        for tick in range(ticks_per_view):
            # check the previous tick's applied forces and see if they are large or small
            # used to dynamically change the time step for every sim tick
            # to speed up those final ever-so-slightly-still-moving moments
            # all values eye-balled
            if len(applied_forces) > 0 and len(spike_applied_forces) > 0:
                last_max_force = max( np.max(np.abs(applied_forces[-1])),
                                      np.max(np.abs(spike_applied_forces[-1])))
                if last_max_force > 0.01:
                    current_dt_step = 0
                elif last_max_force <= 0.01:
                    current_dt_step = 1
                    if last_max_force <= 0.005:
                        current_dt_step = 2

            # update dt
            dt = dt_steps[current_dt_step]

            t0 = time.time()

            # update the swarm
            forces = swarm.get_acting_forces(dt)
            forces = phys.collide_planes(swarm, forces, planes, dt)
            swarm.update(dt, forces=forces)

            # same for the spikes
            # but also update the centers of the spikes so that they
            # follow the swarm
            spikes.set_center(swarm._pos)
            s_forces = spikes.get_acting_forces(dt)
            s_forces = phys.collide_planes(spikes, s_forces, planes, dt)
            s_forces = phys.apply_chain(swarm, spikes, spikes._radius, forces, s_forces, dt)
            spikes.update(dt, forces=s_forces)

            tick_times.append(time.time()-t0)

            # record for later
            edges.append(G.create_cage(swarm._pos))
            traces.append(np.copy(swarm._pos))
            sphere_err.append(swarm.check_sphere())
            applied_forces.append(forces)

            spike_applied_forces.append(s_forces)

        # finally show the state of the swarm
        swarm_view.update()
        spike_view.update()
        rate.sleep()

        # stop if agents converged
        if not swarm._any_alive() and not spikes._any_alive():
            print('All dead. Avg ticks per second:', 1/np.average(tick_times))
            print('Ticks:',len(applied_forces))
            break

    ########################################################################################
    # FINALIZE
    ########################################################################################
    # send the final locations to rviz
    for i in range(50):
        swarm_view.update()
        spike_view.update()
        rate2.sleep()

    # no need to profile plotting
    if profiling:
        sys.exit(0)
    ########################################################################################
    # PLOTTING
    ########################################################################################
    # time, agent, (x,y,z)
    traces = np.array(traces)
    # time, edge, [(x,y,z), (x,y,z)]
    edges_in_time = np.array(edges)
    # time, agent, (u,v,w)
    applied_forces = np.array(applied_forces)

    ########################################################################################
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(sphere_err)
    plt.xlabel('sim ticks')
    plt.ylabel('sphere err')
    plt.title('sphere error')

    edge_lens = []
    for t,edges in enumerate(edges_in_time):
        for edge in edges:
            if edge is not None:
                L = G.euclid_distance(edge[0], edge[1])
                edge_lens.append((t,L))
    edge_lens = np.array(edge_lens)

    plt.subplot(1,2,2)
    # edge_lens might not have the same number of edges at every time step!
    # so we can not really make a 2d array out of it to plot
    # we can, however, scatter what we got
    plt.scatter(edge_lens[:,0], edge_lens[:,1], marker='.', alpha=0.3)
    plt.axhline(1.63, linestyle='--')
    plt.xlabel('sim ticks')
    plt.ylabel('edge lengths of cage')
    plt.ylim(0,3)
    plt.title('cage edges over time')
    ########################################################################################

    fig, ax = P.make_3d_fig()
    P.scatter3(ax, swarm._pos)
    P.scatter3(ax, np.array((0,0,0)), color='g')

    every=200
    for i in range(N):
        P.plot3(ax, traces[::every][:,i,:])
        P.arrows3(ax, traces[::every][:,i,:], applied_forces[::every][:,i,:], length=0.1, normalize=True, color='r', alpha=0.3)

    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)
    plt.title('trajectories of points')




    ########################################################################################
    # plot a sphere, check each point for coverage
    nu, nv = 100,100
    sensor_range = 0.5
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

            dist = G.euclid_distance(xyz, swarm._pos)
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



