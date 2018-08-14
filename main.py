#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-07-26


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


class ConeSwarm:
    def __init__(self, agent_swarm, look_swarm):
        """
        a simple swarm that uses two other swarms for its data
        """

        self.agent_swarm = agent_swarm
        self.look_swarm = look_swarm

    def get_position(self):
        return self.agent_swarm.get_position()

    def get_look_vectors(self):
        look_vectors = self.look_swarm.get_position() - self.agent_swarm.get_position()
        #  look_vectors = self.look_swarm._center - self.agent_swarm.get_position()
        return look_vectors

    def get_orientation_quat(self):
        look_vectors = self.get_look_vectors()
        quats = []
        for vel in look_vectors:
            yaw,pitch = G.vec3_to_yaw_pitch(vel)
            roll = 0

            yaw*= G.RADTODEG
            pitch*= G.RADTODEG
            quats.append(Quat([yaw,pitch,roll]).q)

        return quats



if __name__=='__main__':
    ########################################################################################

    # N, num of agents
    # 4,6,12
    N = 6
    # where the planar obstacles are in u-v-r coords
    plane_uvrs = [ (0,0,0.8), (0,0,-0.8)]#, (2,1,0.6)]#, (1.2, 1.2, -0.6)]
    # dt, time step per sim tick
    # during run, the simulation will change between these when needed
    dt_steps = [0.005, 0.01, 0.05]
    current_dt_step = 0
    # ups, updates per second for ros stuff to update
    ups = 60
    # ticker per view, how many sim ticks to run per view update
    ticks_per_view = 5

    # set to false when not running a profiler
    # this changes the ros init_node disable signal and plotting stuff and saving stuff
    profiling = True
    use_cones = True

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
                                     radius=[0.15]*N,
                                     mass=0.01,
                                     damping=0.05)

    cones = ConeSwarm(agent_swarm = spikes,
                      look_swarm = swarm)

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

    cone_view = RosSwarmView(swarm=cones,
                              mesh_path=mesh_dir+'known_cone.dae',
                              mesh_rgba=(0.5, 0.4, 0.7, 0.4),
                              mesh_scale=(1, 1, 1),
                              last_used_id=spike_view.last_used_id+10)


    # we also want to see a plane and sphere in rviz, representing the cage and obstacle
    # create views for them. Since these are stationary, we wont be updating them much
    # we just need the position of this, it wont be moving
    #  for center, radius in zip(swarm._center, swarm._radius):
    center = swarm._center[0]
    radius = swarm._radius[0]
    sphere_body = VelocityPoint(init_pos=center)
    sphere_view = RosPoseView(body=sphere_body,
                              topic='/sphere_cage')
    marker_array.add_view(sphere_view, mesh_dir+'known_sphere.dae',
                          (radius,radius,radius),
                          (1.0,0.3,0.3,0.4))

    ########################################################################################
    # PLANES
    ########################################################################################
    # points on planes
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

    # to be saved later
    cage_edges = []


    applied_forces = []
    spike_applied_forces = []
    tick_times = []
    elapsed_time = []

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
            # record for later
            cage_edges.append(G.create_cage(swarm._pos))
            applied_forces.append(forces)

            if use_cones:
                # same for the spikes
                # but also update the centers of the spikes so that they
                # follow the swarm
                spikes.set_center(swarm._pos)
                s_forces = spikes.get_acting_forces(dt)
                s_forces = phys.collide_planes(spikes, s_forces, planes, dt)
                spikes.update(dt, forces=s_forces)

                # record
                spike_applied_forces.append(s_forces)

            tick_times.append(time.time()-t0)
            elapsed_time.append(dt)



        # finally show the state of the swarm
        swarm_view.update()

        if use_cones:
            spike_view.update()
            cone_view.update()
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
        if use_cones:
            spike_view.update()
            cone_view.update()
        rate2.sleep()

    # elapsed time should have the 'time' of each update, not the dt
    elapsed_time = np.cumsum(elapsed_time)

    # save all the data we have
    caging_data = {}
    caging_data['caged_sphere'] = (swarm._center, swarm._radius)
    caging_data['obstacle_planes'] = planes

    caging_data['sensor_positions'] = swarm.get_position()
    caging_data['cage_edges_over_time'] = cage_edges

    caging_data['agent_positions'] = spikes.get_position() if use_cones else swarm.get_position()
    caging_data['agent_orientations'] = cones.get_look_vectors() if use_cones else None

    caging_data['sensor_range'] = spikes._radius[0]
    caging_data['elapsed_time'] = elapsed_time

    caging_data['sensor_shape'] = 'conical' if use_cones else 'spherical'

    # make a usable and *'able name for the file
    cone_str = ' '+caging_data['sensor_shape']
    run_string = 'cage '+\
                 '['+str(N)+' '+cone_str+']'+\
                 '['+str(len(planes))+' planes]'+\
                 '['+time.asctime()[4:16]+']'+\
                 '.pickle'

    if not profiling:
        with open(run_string, 'wb') as fout:
            pickle.dump(caging_data, fout)
