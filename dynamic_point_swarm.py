#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-26

import numpy as np
import geometry as G
import Pid
from Quaternion import Quat

RADTODEG = 360 / (np.pi * 2)

class DynamicPointSwarm:
    def __init__(self,
                 init_pos=None,
                 num_agents=4,
                 init_vel=None,
                 mass=1,
                 damping=0.2):
        """
        A collection of points moving about in 3D space with no restrictons

        init_pos and init_vel are list of [x,y,z] or np array of shape (N,3)
        if num_agents is given with init_pos, init_pos is used.
        if nothing is given, 4 agents at 0,0,0 will be created.
        """

        if init_pos is not None:
            # check and assign the initial positions for the particles
            if init_pos.shape[1] != 3:
                raise Exception('Illegal init_pos shape to DynamicSwarm3. Expected (N,3), got '+str(init_pos.shape[1]))

            self._pos = init_pos
        else:
            self._pos = np.zeros((num_agents,3))

        self._num_points = self._pos.shape[0]

        # if given, assign the velocities too, otherwise assume 0
        if init_vel is not None:
            if init_vel.shape[1] != 3:
                raise Exception('Illegal init_vel shape to DynamicSwarm3. Expected (N,3), got '+str(init_vel.shape[1]))
            # given vels
            self._pos = init_pos
        else:
            # not given vels
            self._vel = np.zeros_like(self._pos)

        # all particles have the same mass and damping
        self._mass = mass
        self._damping = damping

    def _any_alive(self, vel_limit=0.001):
        """
        if no particle is moving, retun false, otherwise return true
        """

        alives = np.abs(self._vel) >= vel_limit
        return np.sum(alives) > 1

    def update(self, dt, forces=None, return_positions=False):
        """
        dt is time spent for this update in seconds
        forces is an np array of size (N,3)

        if return_positions is True, returns a copy of the position array
        """

        if forces is None:
            forces = np.zeros((self._num_points,3))
        else:
            if forces.shape != self._vel.shape:
                raise Exception('Force shape not the same as velocities! '+str(forces.shape))


        # apply simple damping
        self._vel *= (1 - self._damping)
        # F = ma
        acc = forces/self._mass
        # v = at
        self._vel += acc*dt
        # x = vt
        self._pos += self._vel*dt

        if return_positions:
            return np.copy(self._pos)

    def get_positions(self):
        return self._pos

    def get_orientation_quats(self):
        # use the velocity as orientation
        # ra=heading/yaw, dec=pitch, roll=roll
        quats =[]
        for vel in self._vel:
            yaw,pitch = G.vec3_to_yaw_pitch(vel)
            roll = 0

            yaw*=RADTODEG
            pitch*=RADTODEG
            quats.append(Quat([yaw,pitch,roll]).q)

        return quats




class DynamicPointSphereSwarm(DynamicPointSwarm):
    def __init__(self,
                 center=[0,0,0],
                 radius=1,
                 charge=1,
                 max_surf_error = 0.3,
                 **kwargs):
        """
        a swarm of points in 3D space with a soft restriction of staying
        on the given sphere
        """
        super().__init__(**kwargs)

        self._center = np.array(center, dtype='float64')
        self._radius = radius
        self._charge = charge

        # if away from the sphere this much, points will ignore others' push
        self._max_surf_error = max_surf_error

        # PID values determined experimentally
        P = 2
        I = 0.01
        D = 0.5
        self._PIDs = [Pid.PID(P,I,D) for i in self._pos]


    def check_sphere(self):
        """
        return the distances to the sphere's surface for all points in swarm
        """

        dists = G.euclid_distance(self._pos, self._center)
        errs = dists - self._radius
        return errs


    def _calc_forces(self, dt):
        """
        calc the net force acting on all points.
        """

        forces = np.zeros_like(self._pos)
        tangent_forces = np.zeros_like(self._pos)

        surface_errors = self.check_sphere()

        for i,this in enumerate(self._pos):
            surface_error = surface_errors[i]

            # this is the normal vector on the sphere, we want the forces to be 0
            # on this vector so that the point does not try to move away from the sphere surface
            _, normal_vec = G.vec_normalize(this - self._center)
            # we always want the point to go towards a point that is ON the sphere.
            # if it is somehow away from the surface, add a force that will push it towards it.
            correction = self._PIDs[i].update(surface_error, dt)
            forces[i] += normal_vec * correction

            # do not apply forces from others onto this particle if it is out of the sphere
            # surface. This makes it so the particles prioritize surface before distribution.
            if np.abs(surface_error) > self._max_surf_error:
                continue


            # add up all the other points' effects on this point.
            for j,other in enumerate(self._pos):
                dist = G.euclid_distance(this, other)
                if dist < 0.01:
                    # disregards self and too-close points
                    continue

                # magnitude of the force is not the vectors norm!
                force_mag = self._charge / (dist**4)
                _, force_vec = G.vec_normalize(this - other)
                force_vec *= force_mag
                perpendicular_vec = G.project_vec(force_vec, normal_vec)
                tangent_vec = force_vec - perpendicular_vec
                tangent_forces[i] += tangent_vec

        # finally add the tangent forces, if any, to the suface forces
        # added previously
        forces += tangent_forces
        return forces

if __name__=='__main__':
    from rosview import RosSwarmView
    import rospy

    N = 12
    ups = 100000
    dt = 0.01

    rospy.init_node('rosviewtest', anonymous=True)
    rate = rospy.Rate(ups)

    init_pos = np.random.random((N,3))
    swarm = DynamicPointSphereSwarm(init_pos=init_pos)

    swarm_view = RosSwarmView(swarm=swarm,
                              mesh_path='/home/ozer/monkey.dae',
                              mesh_rgba=(0.2,0.8,0.2,1),
                              mesh_scale=(0.3,0.3,0.3))


    sphere_err = []
    while not rospy.is_shutdown():
        forces = swarm._calc_forces(dt)
        swarm.update(dt, forces=forces)
        swarm_view.update()
        sphere_err.append(swarm.check_sphere())
        #  rate.sleep()
        if not swarm._any_alive():
            print('all dead')
            break

    uvr_pos = np.array([G.xyz_to_uvr(pos) for pos in swarm.get_positions()])
    import matplotlib.pyplot as plt
    #  plt.ion()
    plt.scatter(uvr_pos[:,0], uvr_pos[:,1])
    plt.figure()
    plt.plot(sphere_err)
    plt.show()
